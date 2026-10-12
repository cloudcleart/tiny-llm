# Week 3 Day 4：直接分页 Attention 实现解析

本次实现把“分页存储”连接到“分页执行”：模型将新 K/V 写入当前层的共享页池，然后把物理页 buffer、页表和长度传给 Metal kernel。整个正常路径不把每个请求的 KV 收集成 dense tensor。已有的分页分配器、缓存事务回滚和模型路由沿用现有实现；本次重点补全算子、embedding，并完善生命周期边界。

## 1. 先明确数据布局和所有权

阅读 `src/tiny_llm/paged_kv_cache.py` 的 `TinyKvPagedCache` 与 `src/tiny_llm/kv_cache.py` 的 `BatchingKvCache.update_and_fetch_paged`。

| 数据 | shape | 含义 |
| --- | --- | --- |
| query | `[B, H_q, L, D]` | 当前计算的 query chunk |
| key_pages / value_pages | `[P, H_kv, page_size, D]` | 当前层的物理页池 |
| block_table | `[B, max_pages]` | 每个请求按逻辑顺序排列的物理页号 |
| context_lens | `[B]` | 写入当前 chunk 后的有效上下文长度 |

一个请求在每一层拥有独立的 cache handle；同一层的所有请求共享一个 pool。页号只在所属层内有意义。批量缓存按调度槽位构造页表，空闲槽位长度为零，页表全部为 `-1`。

`update_and_fetch_paged` 的顺序必须是先追加 K/V，再取得 metadata。追加可能分配新页甚至让 pool 扩容；提前取得 buffer 就可能把旧存储与新页表组合起来。填充原有尾页时页表不变，因此已有的页表缓存可以继续复用。

例如 `page_size=4`、页表 `[12, 5, 3]`、长度 `10`：逻辑 token 0–3 在页 12，4–7 在页 5，8–9 在页 3。页 3 后两个槽位不属于上下文，即使留有其他请求的旧数据也必须忽略。

## 2. Python 边界：把错误挡在 GPU 之前

阅读 `src/tiny_llm/attention.py` 的 `paged_attention`。

入口校验 shape、dtype、GQA 头数关系、页大小、长度、有效页范围、请求内部重复页号，以及未使用页表项的 `-1` sentinel。长度为零表示空闲槽位；其他长度必须至少等于 query 长度。

只支持 `None` 和 `"causal"` 两种 mask。此次没有把任意 additive mask 悄悄转到其他 attention 实现。输入和输出支持 float32、BF16，且 dtype 一致。

Python 把 Q 的前两维折叠成 `[B*H_q,L,D]`，通过 `mx.contiguous` 确保 kernel 可以线性寻址。页池仍保持物理页布局，不按请求排列或复制 K/V。

元数据校验的 `tolist()` 会引入 CPU/GPU 同步。这是可读性优先的校验边界，不能把当前实现当成已经消除全部调度开销的最终性能版本。

## 3. C++：创建 lazy primitive 并启动 Metal

阅读 `src/extensions/src/paged_attention.cpp`。

`tiny_llm_ext::paged_attention` 验证 buffer 形状和类型，创建 MLX lazy 节点。真正分配输出和编码 GPU 命令发生在 `PagedAttention::eval_gpu`。

按照章节要求，C++ 在 `L=8/9` 处选择不同的执行方式：

| 输入条件 | Metal Kernel | Threadgroup 分工 |
| --- | --- | --- |
| `L <= 8`，float32 | `paged_attention_decode_f32` 或宽 D 变体 | 一个 Query 行，32 个 SIMD group，共 1024 个线程 |
| `L <= 8`，BF16 且 `D=128` | `paged_attention_decode_bf16_d128` | 编译期固定 D=128，每个 lane 持有四个连续维度 |
| `L <= 8`，其他 BF16 head dimension | `paged_attention_decode_bf16`、`_max256` 或 `_wide` 变体 | 运行期 D，按范围选择完整累积或分块累积，保持 BF16 输入输出 |
| `L > 8`，float32 / BF16 | `paged_attention_scalar_f32` / `paged_attention_scalar_bf16` | 16 个 Query 行共享 32 个 K/V 位置，共 512 个线程 |

短 Query 的基础通用版本支持 `D <= 128`；`128 < D <= 256` 使用每个 lane 最多八个分量的 `_max256` 实例。`D > 256` 使用 `_wide` 实例，每轮只保存 128 维输出，即每个 lane 四个连续分量。它为每个输出切片重新遍历完整上下文、重新计算完整 D 维 QK 点积，以额外计算换取较低寄存器占用，保持章节要求的 32 个 SIMD group 和 1024 线程。C++ 对 `D > 2048` 明确报错。`D=32/37` 始终按实际维度计算，不会被 BF16 D=128 专用实例重新解释。

例如 D=257 的宽维度 Decode 做三轮输出切片：0–127、128–255、256。每一轮的 Softmax 分数都由全部 257 维 Q/K 计算，不能只对当前 128 维切片做点积；只有 V 加权累积与输出写入被拆分。最后一个不完整切片检查实际 D，避免读写不存在的分量。BF16 D=128 专用路径仍只遍历一次上下文，不使用这种宽维度回退方式。

Decode 只分配一个 32×32 转置 tile 和两个 32 元素状态数组：`(32*32+32+32)*sizeof(float)=4352 B`，即 4.25 KiB，大小不随上下文长度或 D 增长。长 Query 的 staging 则固定为 26,752 B，沿 D 以 128 维分块，避免宽 head dimension 使 threadgroup 内存超过设备容量。

CPU evaluator 明确报告 GPU-only，不悄悄调用 dense 实现。

## 4. Metal：页寻址、可见范围、online softmax

阅读 `src/extensions/src/paged_attention.metal` 的 `paged_attention_decode` 和 `paged_attention_scalar`。两者共用页池与元数据契约，各自采用适合 Query 数量的工作分配。

### 4.1 从逻辑位置换算物理地址

```text
logical_page = token_position // page_size
slot         = token_position % page_size
physical     = block_table[batch, logical_page]
kv_head      = query_head // (H_q // H_kv)
address      = ((physical * H_kv + kv_head) * page_size + slot) * D + d
```

短 Query 按页遍历，页号解析位于 token 循环外。group `g` 处理该页的槽位 `g, g+32, g+64, ...`，复用物理页基址。D 不超过 256 时，每个 lane 提前加载自己负责的 Q 分量，随后通过 `simd_sum` 合并 QK 点积。D=128 时 lane `l` 负责 `4*l + {0,1,2,3}`，Q 与未归一化输出在遍历历史期间保持为局部 FP32 数组。宽维度回退路径用 `lane, lane+32, ...` 分担完整点积，但仍按四个连续分量分配当前输出切片。

长 Query 的一个 threadgroup 负责 16 行 Query，一个 SIMD group 负责其中一行。在计算一个 32-key tile 的分数时，lane `k` 负责该 tile 的第 k 个 key；16 行共用合作加载的 K tile。若 32-key tile 完全位于同一逻辑页内，则整个 tile 共用一次页号解析；跨页 tile 才为每个有效 key 查表。未使用位置写成无效页号，既不读取其页表项，也不访问物理存储。

### 4.2 有历史前缀的 causal mask

当前 Q 对应上下文最后 L 个位置。所以第 i 行 query 的绝对位置为：

```text
absolute_position = context_len - L + i
visible_count     = absolute_position + 1
```

例如历史前缀长度 5、本轮 L=3、写入后总长度 8，三行 query 分别只能读前 6、7、8 个 token。用 `i+1` 会漏掉历史前缀；用 `context_len` 则会让前面的 query 看到未来 token。

短 Query 的尾页取 `min(page_size, visible_count-page_start)`，不会读取该 Query 的未来位置或未写入尾槽。

长 Query 则以当前 16 行中的最后一行确定 tile 的最大可见范围，供多行共享 K/V staging；每行计算分数时再使用自己的 `visible_count` 排除未来位置。范围外的 K/V staging 直接填零，因此未写入的 NaN 尾槽不会被读取。最后一个 Query tile 中不足 16 行时，空行仍参与所有 threadgroup barrier，但不计算有效分数或写入输出。空闲请求没有任何 K/V tile，最终输出零，也不会读取其 `-1` 页号。

### 4.3 每个分区的 online softmax

对已经访问的分数 x 和对应的 V，维护：

```text
m = max(x)
s = sum(exp(x-m))
a = sum(exp(x-m) * V)
```

遇到新分数 x 时更新：

```text
m_new = max(m, x)
alpha = exp(m-m_new)
beta  = exp(x-m_new)
s_new = alpha*s + beta
a_new = alpha*a + beta*V
```

分母和输出分子必须一起重缩放。只调整分母会改变最终结果。

Decode 的 32 个 SIMD group 完成各自历史分区后，先发布最大值与分母，再将所有分区转换到共同最大值 M：

```text
factor_g = exp(m_g-M)
output   = sum(factor_g*a_g) / sum(factor_g*s_g)
```

没有处理任何 token 的分区，其分母为零，合并权重显式取零，避免 `exp(-inf-(-inf))` 产生 NaN。局部输出通过 32×32 tile 转置：写入时索引是 `[dimension_owner][context_partition]`，读出时一个 SIMD group 的 32 个 lane 分别读取 32 个历史分区，用 `simd_sum` 合并同一个输出分量。D=128 复用这个 tile 四轮，而不是在共享内存里保存 32 份完整 D 维向量。每轮读写之间都有完整 threadgroup barrier，防止下一轮覆盖尚未读取的结果。

长 Query 则以 32 个分数为一次 Online Softmax 更新单位。每行先用 `simd_max` 获得 tile 最大分数，再用 `simd_sum` 计算指数权重和，把权重暂存到 16×32 概率 tile。K staging 随后复用为 V staging，各 lane 分担输出维度，累积自己的 FP32 输出分量。

长 Query 的 scratch 组成如下：

| 区域 | 元素 | 字节 |
| --- | --- | ---: |
| Q staging | `16*128` float | 8192 |
| K/V staging，共用一块 buffer | `32*128` float | 16384 |
| 当前 tile 概率 | `16*32` float | 2048 |
| 物理页号 | 32 int | 128 |
| 合计 | | 26,752 |

当 D 超过 128，点积沿 D 分块累加，V 也沿 D 分块合作加载。每个输出分量仍保存完整的 FP32 Online Softmax 累积状态，直到最终归一化才转回输入类型。共享 staging 提供当前 tile 的局部复用；它不是把整个请求 Gather 成 dense K/V，也不创建 `[L,S]` 分数矩阵。Day 5 的 MMA Prefill 不属于此 Day 4 实现。

## 5. 量化 embedding：查表与反量化合并

阅读 `src/tiny_llm/embedding.py` 以及 `src/extensions/src/quantized_matmul.cpp/.metal`。

Week 3 已将 `use_custom_kernel=True` 传给 embedding。本次使这个开关实际调用扩展。每个线程负责一个输出元素，在一个 GPU dispatch 中完成 token 定位、W4 解包、scale/bias 查找和反量化。

```text
pack_index = token_id * (D/8) + d/8
code       = (packed[pack_index] >> (4*(d%8))) & 15
group      = token_id * (D/128) + d/128
```

特别注意数值契约：现有 Python 可读路径先做 BF16 乘法，再做 BF16 加法。因此 kernel 也先将 `code*scale` 舍入到 BF16，再与 bias 相加。改成单次 float FMA 虽可能更精确，却不再复现现有模型的数值轨迹。

无 bias 时传入零 bias；uint32 采样结果转为 int32 token 索引。kernel 对越界 token 输出 NaN，避免访问越界内存。此自定义路径限定 W4、group_size=128；Week 2 默认仍使用原来的可读路径。

## 6. 模型和连续批处理

阅读 `qwen3_week3.py` 的 attention 调用与 `batch.py`。

模型路由原本已接到 `update_and_fetch_paged → paged_attention`，本次补齐了这条路径依赖的实现并添加注释。关闭分页 attention 的显式开关仍可用于 dense 对照实验。传入不支持分页的 cache 会明确报错。

调度器增加 `try/finally`：无论正常结束、prefill 失败、decode 失败还是文本处理失败，都清理 pending prefill 和活跃 batch 的每层缓存。底层 `release` 清空其 page_ids，后续重复调用不会重复归还页面。

还修正了总长度上限：新生成、尚未写入 KV 的 token 已经占据一个输出位置。因此 `offset+1` 达到上限时便结束请求，避免再做一次 KV 写入和无用预测。

## 7. 验证方式与结果

先安装依赖并检查 Metal 环境，再重建扩展：

```bash
pdm install -v
pdm run check-installation
pdm run build-ext
```

课程入口会复制对应的参考测试到 `tests/`。额外边界测试保留在单独文件中，验收时需要显式运行，以免只验证数值结果而漏掉 `L=8/9` 调度边界：

```bash
pdm run test --week 3 --day 4 -- tests/test_week_3_day_4_edges.py -k task_2
pdm run test --week 3 --day 4 -- tests/test_week_3_day_4_edges.py
```

正确性验收应覆盖 L=8/9 的调度边界、BF16 D=128 专用实例、通用 D=32/37 与较宽 head dimension、多页历史前缀、非连续物理页、NaN 尾槽、GQA、空闲 batch 行，以及两种 dtype 和两种 mask。对照使用独立 NumPy float64 方程或可读 Dense Attention，不以测试全绿代替对 Kernel 名称、线程分工和 scratch 配置的源码核对。现有边界文件还包含 scalar/空 token embedding、无 bias、dense cache 拒绝，以及两层小模型连续批处理禁止 dense gather 的测试。

测试使用本地随机小模型，不下载真实权重。尚未执行 Qwen3-4B 性能评估，不能据此声称快于 dense 或 MLX。当前仓库的 `bench-week3-attention` 仍固定导入参考实现，也不接受在线文档中的 `--solution` 参数；直接照搬网页命令无法衡量本次 learner 实现，后续性能实验需先更新 runner 的方案选择。

本次调度修改已重新执行依赖安装、环境检查和扩展构建，均成功。2026-10-10 的首次验收以及 2026-10-12 重建扩展后的提交前复核均为 **105 passed**：课程测试 37 项，补充边界测试 68 项。补充测试实际覆盖短 Query 的 L=1/8、长 Query 的 L=9/17/65，以及 D=32/37/128/129/257/1025/2046；同时覆盖上述非连续物理页、NaN 尾槽、GQA 和空闲行。新增 NumPy 对照直接基于输入数组计算，独立于 Metal 调度实现。

`ruff check tests/test_week_3_day_4_edges.py` 和 `git diff --check` 也通过。源码核对确认短 Query 启动 32 个 SIMD group、BF16 D=128 选用专用实例与 4352 字节 scratch，长 Query 使用独立的分页 tile 调度。此轮未进行速度对比，因此这些结果证明正确性和章节要求的实现结构，不代表已经测得性能提升。
