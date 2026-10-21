# Week 3 Day 5：Paged FlashAttention 验证记录

本目录保存 Day 4 标量分页 Attention 与 Day 5 最终 BF16 MMA 实现的算子对比。Day 5 使用 64 行 query、32 行 K/V 的分块和课程自有 SIMD-matrix 指令，并对固定 fragment、归约和加载循环进行 full-unroll。模型接口、分页 KV 存储和输出保持 BF16；`L <= 8` 的 decode 分发与内核未改。

## 环境与原始文件

- GPU：Apple M1，架构 `applegpu_g13g`；统一内存 8 GiB。
- MLX：0.32.0。
- macOS：26.6.2，build 25G83。
- [operator-day4.json](operator-day4.json)：`day4-scalar`，修改前的算子结果。
- [operator-day5.json](operator-day5.json)：`day5-mma-final`，最终 full-unroll BF16 MMA 的结果。
- [operator_bench.py](operator_bench.py)：输入构造、误差检查及同步计时脚本。

两份 JSON 保留全部原始样本，以微秒记录时间；下表换算为毫秒。原始记录没有因格式清理而重新生成。

## 方法与计时边界

固定 `B=1`、`Hq=32`、`Hkv=8`、`D=128`、`page_size=128`，使用 BF16 输入与 causal mask。`L` 是本次 query 长度，`S` 是包含本次 query 的总上下文长度，因此 causal 的历史前缀长度为 `S-L`。

每种形状重新创建 NumPy RNG，种子为 613。脚本生成 Q/K/V，把逻辑 K/V 切页并按随机排列放入物理页，再用对应页表调用公开的 Python `paged_attention` API。各形状的上下文均为完整页；部分页、非连续页与其他边界由回归测试另外覆盖。

每次调用后立即 `mx.eval(out)`。计时包含 Python API 的 metadata 值校验、同步、分发、输出分配和 GPU 执行；这不是纯 Metal kernel 时间。输入构造、参考结果计算和误差检查在计时之外。

每种形状先运行一次并与 `mx.fast.scaled_dot_product_attention` 比较，再额外预热 5 次，最后在**单个进程中同步运行 15 次，报告这 15 个样本的中位数**。每种形状结束后调用 `mx.clear_cache()`，不计入该形状的样本时间。两份记录不是多进程重复实验，也没有提供置信区间。

## 算子结果

延迟比为 `Day 4 中位数 / Day 5 中位数`；大于 1 表示该形状的 API 调用中位数更低。最后一列是 Day 5 随机输入相对 MLX 参考结果的最大绝对差。

| L | S | Day 4 中位数 / ms | Day 5 中位数 / ms | 延迟比 | Day 5 最大绝对差 |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 3.966 | 1.002 | 3.96× | 0.0078125 |
| 128 | 1024 | 51.420 | 2.211 | 23.25× | 0.001953125 |
| 512 | 1024 | 160.641 | 6.433 | 24.97× | 0.001953125 |
| 1024 | 1024 | 213.743 | 14.676 | 14.56× | 0.0078125 |
| 512 | 8192 | 1661.175 | 104.283 | 15.93× | 0.00048828125 |
| 8 | 1024 | 1.889 | 2.632 | 0.72× | 0.00006103515625 |
| 1 | 8192 | 2.025 | 2.286 | 0.89× | 0.000244140625 |

五种长查询的 API 中位数下降。最后两行仍使用同一个 Day 4 decode 路径，记录却有波动：分别高约 39.3% 与 12.9%。这些单进程样本不能证明 decode 算法发生了加速或稳定回退；需要独立重复测量才能判断稳定性。Day 5 的 `L=1024,S=1024` 样本也存在随测量顺序增长的现象，完整序列见 JSON。

以上倍数仅适用于这组输入、设备和计时边界，不能当成 Qwen 模型、prefill 服务或端到端系统的加速倍数。误差列只覆盖这里的随机输入，不替代边界回归。

## 复现算子计时

从仓库根目录执行，先构建目标阶段的扩展，再启动一个新进程：

```bash
pdm run build-ext
pdm run python benchmark_results/week3-day5-20261016/operator_bench.py \
  --stage day5-mma-final --iterations 15 \
  --output /tmp/week3-day5-operator-recheck.json
```

`--stage` 只为 JSON 和控制台结果添加标签，不切换内核。复现 `day4-scalar` 需要先使用修改前的 Day 4 源码构建扩展，再以相同脚本和参数启动新进程，并把 `--stage` 改为 `day4-scalar`。不能在当前 Day 5 二进制上更改标签来重建 Day 4 基线。`--limit` 可截取固定形状列表的前几项；默认运行全部七项。

脚本按自身文件位置找到仓库根和 `src`，无需额外设置 `PYTHONPATH`。`--help` 不加载 MLX，也不会执行 GPU 计算。

## 正确性验证

最终 BF16 MMA 实现已复跑 **135 项测试，全部通过（5.88 秒）**。构建后使用以下目标运行 Day 4/5 的规范测试与边界回归：

```bash
pdm run build-ext
pdm run test --week 3 --day 5 -- \
  tests/test_week_3_day_5_edges.py \
  tests/test_week_3_day_4.py \
  tests/test_week_3_day_4_edges.py
```

runner 自动补入 `tests/test_week_3_day_5.py`。这组测试覆盖单页/跨页及不连续物理页、query/KV 尾块、未写槽位 NaN、idle batch、GQA、因果历史偏移、自定义 scale、较晚出现最大分数时的历史重缩放，以及 prefill 后继续 decode；同时检查 float32、`D != 128` 与短查询通用路径的兼容性。

## Serving 测量

真实服务结果保存在 [serving-0.6b.json](serving-0.6b.json)。使用本地缓存的 Qwen3-0.6B-MLX-4bit，`solution=tiny_llm`、`loader=week3`、分页大小 128。模型头维度为 128，Query/KV 头数为 16/8，长 BF16 Prefill 自动进入本次实现的 MMA 分发。量化投影沿用已有 MLX seam，缓存、Attention 与调度使用课程实现。

负载为 8 条请求，decode batch size 2，prompt 长度范围 128–1024，output 长度范围 8–16，prefill chunk 128。每个独立进程预热一次后重置页池，再运行同一请求轨迹；共重复 2 个进程，报告进程结果的中位数。配置的 batch size 限制 decode 槽位；另有一个进行中的 prefill 请求，因此记录的峰值活动请求数为 3。

| 指标 | 本机测量 |
|---|---:|
| Prefill 吞吐 | 708.81 tok/s |
| Output 吞吐（含调度及 prefill 时间） | 10.90 tok/s |
| Decode 吞吐 | 57.16 tok/s |
| 请求吞吐 | 0.831 requests/s |
| 峰值 KV 存储 | 448 MiB |
| 逻辑 KV 拷贝量 | 392 MiB |
| Dense staging 拷贝量 | 0 MiB |
| Decode 调用延迟 median / p95 / max | 30.93 / 36.79 / 37.75 ms |

逻辑拷贝量来自页池扩容，记录的 dense growth 与 dense staging 均为 0。页池有 756 次复用分配，尾页浪费占 live-page 槽位的 12.8%。这些计数与原始请求轨迹、每进程结果一起保存在 JSON 中。

```bash
pdm run bench-serving-progression --offline --solution tiny_llm \
  --variant paged --repeats 2 --model qwen3-0.6b \
  --num-seqs 8 --batch-size 2 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 8 --max-output-len 16 --prefill-step 128 \
  --warmup 1 --cooldown-seconds 1 \
  --json-output /tmp/week3-day5-serving-recheck.json
```

书中 Qwen3-4B、16 请求、batch size 4、4 进程重复的配置已尝试，但未得到完整结果，因此不报告其吞吐。上述 0.6B 负载用于本机集成验证，与书中的 4B 配置不同；本次也没有配对运行 Day 4 的同模型服务基线，不能据此给出 Day 5 的端到端加速倍数。8K 上下文已在独立算子测量中覆盖，尚未测量完整 8K 模型的静态吞吐。
