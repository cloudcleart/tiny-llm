#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

// Builds the lazy MLX operation. The actual CPU work happens later in
// QuantizedMatmul::eval_cpu when MLX evaluates the returned array.
mx::array quantized_matmul(const mx::array &scales,   // Per-group dequantization scales, shape (K, N / group_size).
                           const mx::array &biases,   // Per-group dequantization biases, same shape as scales.
                           const int group_size,      // Number of dense weights covered by one scale/bias pair.
                           const int bits,            // Quantization bit width. Task 2 supports int4.
                           const mx::array &a,        // Dense activation matrix, shape (M, N).
                           const mx::array &b,        // Packed quantized weights, shape (K, N / 8) for int4.
                           const bool transpose_b,    // Linear layers use true: output = a @ dequantize(b).T.
                           mx::StreamOrDevice s) {
    // Task 2 intentionally targets the format used by the course weights:
    // bf16 activations/scales/biases, uint32-packed int4 weights, and G=128.
    if (scales.dtype() != mx::bfloat16) {
        throw std::runtime_error("quantized_matmul: scales must be bfloat16");
    }
    if (biases.dtype() != scales.dtype()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same dtype");
    }
    if (a.dtype() != scales.dtype()) {
        throw std::runtime_error("quantized_matmul: a must have the same dtype as scales");
    }
    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: b must be uint32");
    }
    if (a.ndim() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    if (b.ndim() != 2) {
        throw std::runtime_error("quantized_matmul: b must be a 2D array");
    }
    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    if (group_size != 128) {
        throw std::runtime_error("quantized_matmul: group_size must be 128");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: transpose_b must be true");
    }

    const int packs_per_uint32 = 32 / bits;
    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];

    // With transpose_b=true, dense b would have shape (K, N), so the output is
    // (M, K). The packed storage keeps each row of b in N / 8 uint32 values.
    if (N % group_size != 0) {
        throw std::runtime_error("quantized_matmul: a columns must be divisible by group_size");
    }
    if (b.shape()[1] != N / packs_per_uint32) {
        throw std::runtime_error("quantized_matmul: b must pack one row of a per output row");
    }
    if (scales.shape()[0] != K) {
        throw std::runtime_error("quantized_matmul: scales rows must match b rows");
    }
    if (scales.shape()[1] != N / group_size) {
        throw std::runtime_error("quantized_matmul: scales must have one column per input group");
    }

    return mx::array({M, K}, a.dtype(), std::make_shared<QuantizedMatmul>(to_stream(s)), {scales, biases, a, b});
}

// CPU reference kernel for C = A @ dequantize(B).T.
//
// Inputs:
//   A:      (M, N) bf16
//   B:      (K, N / 8) uint32, 8 int4 weights per uint32
//   scales: (K, N / 128) bf16
//   biases: (K, N / 128) bf16
// Output:
//   C:      (M, K) bf16
void quantized_matmul_impl(const mx::array &scales, const mx::array &biases, const mx::array &a, const mx::array &b,
                           mx::array &out, mx::Stream stream) {
    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be row contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be row contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    // Register arrays with MLX's CPU encoder so their lifetimes and execution
    // dependencies are tracked correctly by the stream scheduler.
    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    encoder.dispatch([out_ptr = out.data<mx::bfloat16_t>(), out_shape = out.shape(), out_strides = out.strides(),
                      scales = mx::array::unsafe_weak_copy(scales), biases = mx::array::unsafe_weak_copy(biases),
                      a = mx::array::unsafe_weak_copy(a), b = mx::array::unsafe_weak_copy(b)]() {
        const int M = a.shape()[0];
        const int N = a.shape()[1];
        const int K = b.shape()[0];
        const int group_size = 128;
        const int bits = 4;
        const int values_per_pack = 32 / bits;
        const int groups_per_row = N / group_size;
        const uint32_t item_mask = (1u << bits) - 1u;

        const auto *scales_ptr = scales.data<mx::bfloat16_t>();
        const auto *biases_ptr = biases.data<mx::bfloat16_t>();
        const auto *a_ptr = a.data<mx::bfloat16_t>();
        const auto *b_ptr = b.data<uint32_t>();

        // One scalar output C[i, k] is computed at a time. Accumulation stays
        // in float32 to avoid losing too much precision while summing N terms.
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float sum = 0.0f;

                // Each group of 128 consecutive weights has its own affine
                // dequantization parameters: value ~= q * scale + bias.
                for (int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
                    const int64_t scale_loc = mx::elem_to_loc(k * groups_per_row + group_idx, scales.shape(), scales.strides());
                    const int64_t bias_loc = mx::elem_to_loc(k * groups_per_row + group_idx, biases.shape(), biases.strides());
                    const float scale = static_cast<float>(scales_ptr[scale_loc]);
                    const float bias = static_cast<float>(biases_ptr[bias_loc]);

                    const int group_start = group_idx * group_size;
                    int64_t a_loc = mx::elem_to_loc(i * N + group_start, a.shape(), a.strides());
                    int64_t b_loc = mx::elem_to_loc(k * (N / values_per_pack) + group_start / values_per_pack, b.shape(), b.strides());

                    // Each uint32 stores 8 int4 values in little-endian nibble
                    // order: bits [3:0] are the first weight, [7:4] the second.
                    for (int item_idx = 0; item_idx < group_size; item_idx += values_per_pack) {
                        const uint32_t packed = b_ptr[b_loc++];
                        for (int pack_idx = 0; pack_idx < values_per_pack; ++pack_idx) {
                            const uint32_t q = (packed >> (pack_idx * bits)) & item_mask;
                            const float dequantized_b = static_cast<float>(q) * scale + bias;
                            sum += static_cast<float>(a_ptr[a_loc++]) * dequantized_b;
                        }
                    }
                }

                const int64_t out_loc = mx::elem_to_loc(i * K + k, out_shape, out_strides);
                out_ptr[out_loc] = static_cast<mx::bfloat16_t>(sum);
            }
        }
    });
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];
    quantized_matmul_impl(scales, biases, a, b, out, stream());
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
#ifdef _METAL_
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be row contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be row contiguous");
    }

    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];
    if (N % 128 != 0) {
        throw std::runtime_error("quantized_matmul: N must be divisible by group_size");
    }

    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // Kernel name matches the instantiation in quantized_matmul.metal.
    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel("quantized_matmul_w4a16_g128_bf16", library);

    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    // Two-dimensional launch: x covers M rows, y covers K output features.
    const int x_size = 32;
    const int y_size = std::max<int>(1, kernel->maxTotalThreadsPerThreadgroup() / x_size);
    MTL::Size threads_per_group = MTL::Size(x_size, y_size, 1);
    MTL::Size threadgroups = MTL::Size((M + x_size - 1) / x_size, (K + y_size - 1) / y_size, 1);
    compute_encoder.dispatch_threadgroups(threadgroups, threads_per_group);
#else
    throw std::runtime_error("QuantizedMatmul has no GPU implementation yet.");
#endif
}

}  // namespace tiny_llm_ext
