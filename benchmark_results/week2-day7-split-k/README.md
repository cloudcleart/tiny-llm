# Day 7 Split-K operator validation

Task 3 selects aligned partitions from output-grid size, dispatches private planes, and preserves matvec/unsplit fallbacks. Task 4 merges planes in FP32. Public interfaces are unchanged.

Environment: Apple M1, MLX 0.32.0. See operators.json for full metadata and raw samples.

Measurement: Qwen3-4B, BF16, 5 warmup rounds, 30 timed rounds, rotated implementation order, forward/reverse context sweeps. Values below are aggregated synchronized medians in microseconds.

| Tokens | Projection | Day 6 | Split-K | MLX | Day 6 / Split-K |
|---:|---|---:|---:|---:|---:|
| 16 | prefill q matmul | 1786.4 | 1774.6 | 1074.7 | 1.007 |
| 16 | prefill k matmul | 821.4 | 680.0 | 506.1 | 1.208 |
| 16 | prefill v matmul | 894.6 | 767.9 | 559.7 | 1.165 |
| 16 | prefill o matmul | 1969.1 | 1800.2 | 1118.9 | 1.094 |
| 16 | prefill gate matmul | 3801.2 | 3773.4 | 2089.7 | 1.007 |
| 16 | prefill up matmul | 3798.7 | 3807.4 | 2115.5 | 0.998 |
| 16 | prefill down matmul | 3982.9 | 3824.2 | 2120.6 | 1.042 |
| 32 | prefill q matmul | 1660.2 | 1649.6 | 1138.8 | 1.006 |
| 32 | prefill k matmul | 751.2 | 674.6 | 537.9 | 1.114 |
| 32 | prefill v matmul | 926.7 | 863.6 | 704.4 | 1.073 |
| 32 | prefill o matmul | 1903.5 | 1784.6 | 1229.9 | 1.067 |
| 32 | prefill gate matmul | 3183.1 | 3193.3 | 2017.0 | 0.997 |
| 32 | prefill up matmul | 3152.4 | 3178.2 | 1994.1 | 0.992 |
| 32 | prefill down matmul | 3474.9 | 3333.2 | 2170.2 | 1.043 |
| 64 | prefill q matmul | 2850.3 | 2868.3 | 1884.5 | 0.994 |
| 64 | prefill k matmul | 1102.9 | 1023.3 | 776.7 | 1.078 |
| 64 | prefill v matmul | 1336.6 | 1213.6 | 890.8 | 1.101 |
| 64 | prefill o matmul | 2902.6 | 2867.5 | 1919.4 | 1.012 |
| 64 | prefill gate matmul | 6025.1 | 5952.4 | 3656.4 | 1.012 |
| 64 | prefill up matmul | 6177.7 | 6182.6 | 3798.7 | 0.999 |
| 64 | prefill down matmul | 6416.1 | 6338.5 | 3971.2 | 1.012 |
| 128 | prefill q matmul | 5068.9 | 5062.9 | 3077.8 | 1.001 |
| 128 | prefill k matmul | 1532.6 | 1496.2 | 1007.5 | 1.024 |
| 128 | prefill v matmul | 1625.1 | 1605.4 | 1122.7 | 1.012 |
| 128 | prefill o matmul | 5192.7 | 5232.7 | 3260.1 | 0.992 |
| 128 | prefill gate matmul | 11694.7 | 11688.5 | 7096.0 | 1.001 |
| 128 | prefill up matmul | 11615.2 | 11617.8 | 6983.9 | 1.000 |
| 128 | prefill down matmul | 11838.8 | 11816.9 | 7052.7 | 1.002 |
| 2048 | prefill q matmul | 74463.2 | 74493.4 | 43809.9 | 1.000 |
| 2048 | prefill k matmul | 19443.6 | 19370.5 | 11428.7 | 1.004 |
| 2048 | prefill v matmul | 19285.9 | 19262.3 | 11413.5 | 1.001 |
| 2048 | prefill o matmul | 74679.6 | 74716.9 | 43753.0 | 1.000 |
| 2048 | prefill gate matmul | 175926.0 | 175945.2 | 103135.9 | 1.000 |
| 2048 | prefill up matmul | 176107.6 | 176005.7 | 103149.6 | 1.001 |
| 2048 | prefill down matmul | 177262.4 | 177100.7 | 102925.1 | 1.001 |

At 32 tokens, K projection latency decreases from 751.2 to 674.6 us (about 10.2%). At 2048 tokens, every projection uses the original unsplit kernel; small timing differences reflect measurement variation. Do not compare this run directly with the older Task 1 absolute timings: use the paired Day 6 measurements in this run.

Validation: extension rebuilt; Day 6: 38 passed, 6 skipped. Day 7 also covers BF16 partial tiles, exact fallback, FP16 partial tiles, and exact one-token decode preservation.

This report measures operators only; no complete-model throughput claim is made.

Reproduce:
```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b --section prefill-projections --context 16 --context 32 --context 64 --context 128 --context 2048 --prefill-projection q --prefill-projection k --prefill-projection v --prefill-projection o --prefill-projection gate --prefill-projection up --prefill-projection down --include-split-k --warmup 5 --iterations 30 --json-output benchmark_results/week2-day7-split-k/operators.json
```
