# Week 2 Day 7 Task 1：未切分 K 投影基线

## 测量结果

2026-09-14 执行；源码提交 `03a394b04fa010db93fa89b207e93fbf363a84fc`，测量时工作区干净。

| 实现 | 同步延迟中位数 |
|---|---:|
| Day 6 SIMD-group matmul（未切分） | 1954.958 µs |
| MLX quantized matmul | 1277.479 µs |

Day 6 / MLX 延迟比为 **1.5303×**。本次仅建立优化前基线，没有启用 Split-K 或修改计算函数。

## 形状与环境

- 模型：`Qwen/Qwen3-4B-MLX-4bit`，第一层 K 投影。
- 激活 `[32, 2560]`，权重逻辑形状 `[1024, 2560]`，输出 `[32, 1024]`。
- BF16 激活和量化参数；权重 INT4，group size 128。
- Day 6 输出 tile 为 32×32，启动 `ceil(32/32) * ceil(1024/32) = 32` 个线程组。
- Apple M1，7 核 GPU，8 GB 统一内存；macOS 26.6.2。
- Python 3.12.10，MLX 0.32.0，mlx-lm 0.31.3，Xcode 26.5。

32 个输出线程组提供了后续沿归约维度增加并行任务的机会；本次没有采集 GPU occupancy，不能单凭延迟比证明差距全部来自占用率。页面中的 320 线程组调优目标也不是本机硬件属性。

## 复现

环境准备已成功执行 `pdm install -v` 和 `pdm run check-installation`。

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section prefill-projections --context 32 --prefill-projection k \
  --warmup 5 --iterations 30 \
  --json-output benchmark_results/week2-day7-task1/baseline.json
```

脚本每次通过 `mx.eval(function())` 等待计算完成，计时包含主机调用与同步开销，并非纯 GPU kernel 时间。预热 5 轮，测量 30 轮；两种实现的先后顺序交替，各种顺序各 15 轮。

[baseline.json](baseline.json) 保存全部原始样本、执行顺序、配置、硬件和软件元数据。Day 6 样本范围约为 1191–2547 µs，MLX 为 886–2569 µs，存在明显波动。这是一轮基线，不是多轮稳定性结论；后续比较应保持相同环境和计时方式，并保留此文件。

Task 1 为测量任务，本次未运行 Day 7 的实现测试；分区计算、调度和归约属于后续任务。
