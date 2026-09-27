# Week 1 → Week 2 cumulative performance

This report compares the complete Week 1 model with every cumulative Week 2
checkpoint in the student implementation. Each row contains all features from
the rows above it.

## Method

- Hardware: Apple M1, 7-core GPU, 8 GB unified memory.
- Software: Python 3.12.10, MLX 0.32.0.
- Model: Qwen3-0.6B.
- Workload: 128 input tokens, 64 generated tokens (`output_len=65`).
- Prompt logits: all 128 rows, required for a matched Week 1 comparison.
- Sampling: 2 warmups and 4 measured fresh processes per checkpoint.
- Ordering: alternating forward/reverse checkpoint order to reduce systematic
  thermal bias.
- Values: median throughput in tokens/s; higher is better.

```bash
pdm run bench-week2-progression --offline --solution tiny_llm \
  --suite week2 --model qwen3-0.6b \
  --input-len 128 --output-len 65 --prefill-logits all \
  --warmup 2 --repeats 4 \
  --json-output benchmark_results/week1-week2-progression/qwen3-0.6b-input128-output65.json
```

## Results

| Checkpoint | Feature added | Prefill | Change from previous | Decode | Change from previous | End-to-end output |
|---|---|---:|---:|---:|---:|---:|
| Week 1 | Readable model | 95.44 | baseline | 2.08 | baseline | 2.01 |
| 2.1 | KV Cache | 415.90 | +335.8% | 26.12 | +1153.0% | 23.53 |
| 2.3 | Quantized MatVec | 17.62 | -95.8% | 43.24 | +65.5% | 7.44 |
| 2.4a | Fast RMSNorm | 17.38 | -1.4% | 45.03 | +4.1% | 7.38 |
| 2.4b | Fast RoPE | 17.89 | +2.9% | 52.33 | +16.2% | 7.76 |
| 2.4c | Fused SwiGLU | 17.83 | -0.3% | 56.72 | +8.4% | 7.80 |
| 2.5 | Fused Decode Attention | 17.88 | +0.3% | 61.02 | +7.6% | 7.88 |
| 2.6 | SIMD-Matrix Prefill | 288.06 | +1511.0% | 59.16 | -3.0% | 42.56 |
| 2.7 | Split-K Prefill | 285.76 | -0.8% | 63.79 | +7.8% | 44.84 |
| MLX | Library baseline | 352.42 | +23.3% | 66.05 | +3.5% | 49.12 |

The apparent decode gain in the Split-K row is measurement variation: Split-K
does not run for one-token decode, which retains the MatVec path. Likewise, the
small prefill differences among RMSNorm, RoPE, SwiGLU, and Decode Attention are
not evidence of meaningful prefill changes.

## Interpretation

KV Cache is the largest decode improvement. It changes autoregressive
generation from repeatedly recomputing the full prefix to processing one new
token against cached keys and values. Decode improves 12.53× over Week 1.

Quantized MatVec reduces weight traffic and improves decode by another 65.5%.
Its prefill regression is expected at this checkpoint: the model now uses the
course quantized projection kernel, while the efficient tiled prefill kernel
has not been introduced. Computing all 128 prompt logits also includes the
large vocabulary projection, making the untiled quantized matrix path the
dominant bottleneck.

The three fused model kernels improve decode cumulatively. RMSNorm contributes
4.1%, RoPE 16.2%, and SwiGLU 8.4% in this run. Fused Decode Attention adds 7.6%
by replacing materialized attention intermediates with a dedicated online
softmax kernel.

SIMD-Matrix is the decisive prefill recovery: throughput rises from 17.88 to
288.06 tok/s, a 16.11× increase. It replaces per-output quantized dot products
with cooperative 32×32 tiles and threadgroup data reuse.

Split-K targets under-filled short-prompt projection grids. At this matched
128-token, 0.6B, all-logits workload it is neutral for aggregate prefill
(-0.8%). This does not conflict with the operator results on Qwen3-4B, where
the 32-token K projection improved by about 10.2%; the optimization is
shape-dependent and falls back to SIMD-Matrix when the base grid is occupied.

The final checkpoint reaches 81.1% of MLX prefill throughput, 96.6% of MLX
decode throughput, and 91.3% of MLX end-to-end output throughput on this
workload. Relative to Week 1 it is 2.99× faster in prefill, 30.59× faster in
decode, and 22.31× faster end to end.

## Checkpoint mapping

Week 2 Day 2 adds benchmarking and profiling infrastructure rather than a
runtime feature, so it has no distinct checkpoint. Day 4 contributes three
independently measurable cumulative kernels: RMSNorm, RoPE, and SwiGLU.

The JSON file in this directory contains all raw samples, medians, execution
order, source revision, configuration, and host metadata.
