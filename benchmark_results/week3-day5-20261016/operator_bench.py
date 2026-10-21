"""Measure the public paged-attention API with synchronized synthetic inputs.

Run from the repository root, using the extension binary for the named stage.
Changing --stage labels the output; it does not select a kernel implementation.
"""

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter


SHAPES = (
    (128, 128),
    (128, 1024),
    (512, 1024),
    (1024, 1024),
    (512, 8192),
    (8, 1024),
    (1, 8192),
)
WARMUP = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--limit", type=int, default=len(SHAPES))
    return parser.parse_args(argv)


def benchmark(args):
    # Keep imports here so --help works without initializing MLX. main() adds
    # the checkout's source directory before importing the model-facing API.
    import mlx.core as mx
    import numpy as np

    from tiny_llm.attention import paged_attention

    results = []
    for length, context in SHAPES[: args.limit]:
        rng = np.random.default_rng(613)
        page_size, q_heads, kv_heads, dim = 128, 32, 8, 128
        count = (context + page_size - 1) // page_size
        pages = rng.permutation(count)
        q = mx.array(rng.normal(size=(1, q_heads, length, dim)), dtype=mx.bfloat16)
        key = mx.array(rng.normal(size=(1, kv_heads, context, dim)), dtype=mx.bfloat16)
        value = mx.array(
            rng.normal(size=(1, kv_heads, context, dim)), dtype=mx.bfloat16
        )

        # Split logical K/V into pages, then physically shuffle those pages.
        # The fixed shape list has contexts divisible by page_size.
        logical_k = (
            key.reshape(1, kv_heads, count, page_size, dim)
            .transpose(2, 1, 3, 4, 0)
            .reshape(count, kv_heads, page_size, dim)
        )
        logical_v = (
            value.reshape(1, kv_heads, count, page_size, dim)
            .transpose(2, 1, 3, 4, 0)
            .reshape(count, kv_heads, page_size, dim)
        )
        inverse = mx.array(np.argsort(pages), dtype=mx.int32)
        kp = mx.contiguous(logical_k[inverse])
        vp = mx.contiguous(logical_v[inverse])
        table = mx.array(pages[None], dtype=mx.int32)
        lengths = mx.array([context], dtype=mx.int32)
        mx.eval(q, key, value, kp, vp, table, lengths)

        def run():
            out = paged_attention(q, kp, vp, table, lengths, page_size, mask="causal")
            mx.eval(out)
            return out

        actual = run()
        expected = mx.fast.scaled_dot_product_attention(
            q, key, value, scale=dim**-0.5, mask="causal"
        )
        error = float(
            mx.max(
                mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            ).item()
        )
        for _ in range(WARMUP):
            run()
        times = []
        for _ in range(args.iterations):
            start = perf_counter()
            run()
            times.append((perf_counter() - start) * 1e6)
        row = {
            "L": length,
            "S": context,
            "median_us": median(times),
            "samples_us": times,
            "max_abs_error_vs_mlx": error,
        }
        results.append(row)
        print(
            args.stage,
            length,
            context,
            round(row["median_us"], 2),
            error,
            flush=True,
        )
        mx.clear_cache()

    return {
        "stage": args.stage,
        "mlx": mx.__version__,
        "device": mx.device_info(),
        "shape": {
            "B": 1,
            "Hq": 32,
            "Hkv": 8,
            "D": 128,
            "page_size": 128,
            "dtype": "BF16",
            "mask": "causal",
            "physical_pages": "permuted",
        },
        "method": {
            "warmup": WARMUP,
            "iterations": args.iterations,
            "synchronized": True,
            "timing_boundary": (
                "Python paged_attention API including metadata validation and dispatch"
            ),
            "model_required": False,
        },
        "results": results,
    }


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(repo_root / "src"), str(repo_root)]
    payload = benchmark(args)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
