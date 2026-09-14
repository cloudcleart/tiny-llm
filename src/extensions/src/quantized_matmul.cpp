#include <memory>
#include <stdexcept>
#include <string>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 2, Day 3. Days 6 and 7 extend the dispatch policy behind this API.
mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, int group_size, int bits,
                           const mx::array &a, const mx::array &b, bool transpose_b, bool use_simdgroup,
                           bool use_split_k, mx::StreamOrDevice s) {
    if (scales.dtype() != mx::float16 && scales.dtype() != mx::bfloat16) {
        throw std::runtime_error("quantized_matmul: scales must be float16 or bfloat16");
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
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("quantized_matmul: a and b must be 2D arrays");
    }
    if (scales.ndim() != 2 || biases.ndim() != 2) {
        throw std::runtime_error("quantized_matmul: scales and biases must be 2D arrays");
    }
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    if (group_size != 128) {
        throw std::runtime_error("quantized_matmul: group_size must be 128");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed");
    }
    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }

    const int values_per_pack = 32 / bits;
    const int input_width = a.shape()[1];
    if (input_width % group_size != 0) {
        throw std::runtime_error("quantized_matmul: a columns must be divisible by group_size");
    }
    if (b.shape()[0] != scales.shape()[0]) {
        throw std::runtime_error("quantized_matmul: b must have the same number of rows as scales");
    }
    if (scales.shape()[1] != input_width / group_size) {
        throw std::runtime_error("quantized_matmul: scales must have one column per input group");
    }
    if (b.shape()[1] != input_width / values_per_pack) {
        throw std::runtime_error("quantized_matmul: packed b width does not match a columns");
    }

    // AB^T produces one output column per packed weight row. Constructing the
    // array records a lazy MLX graph node; Task 3 supplies its GPU evaluator.
    auto out_shape = a.shape();
    out_shape[1] = b.shape()[0];
    return mx::array(out_shape, a.dtype(), std::make_shared<QuantizedMatmul>(to_stream(s), use_simdgroup, use_split_k),
                     {scales, biases, a, b});
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("quantized_matmul: the course extension is GPU-only");
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    if (!scales.flags().row_contiguous || !biases.flags().row_contiguous || !a.flags().row_contiguous ||
        !b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: all inputs must be row-contiguous");
    }

    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];
    if (N % 128 != 0) {
        throw std::runtime_error("quantized_matmul: N must be divisible by group_size");
    }

    auto &s = stream();
    auto &device = mx::metal::device(s.device);
    auto library = device.get_library("tiny_llm_ext");
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // Preserve the explicit vanilla oracle; the optimized schedule dispatches
    // tiny M to Day 3 MatVec and prefill to the Day 6 cooperative matrix tile.
    const bool use_matvec = use_simdgroup_ && M <= 8;
    const bool use_tiled = use_simdgroup_ && M > 8;
    const char *kernel_name;
    if (use_matvec) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matvec_x4_fast_w4a16_g128_f16"
                                                 : "quantized_matvec_x4_fast_w4a16_g128_bf16";
    } else if (use_tiled) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_simdgroup_w4a16_g128_f16"
                                                 : "quantized_matmul_simdgroup_w4a16_g128_bf16";
    } else {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_vanilla_w4a16_g128_f16"
                                                 : "quantized_matmul_vanilla_w4a16_g128_bf16";
    }

    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = mx::metal::get_command_encoder(s);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(scales, 0);
    encoder.set_input_array(biases, 1);
    encoder.set_input_array(a, 2);
    encoder.set_input_array(b, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(M, 5);
    encoder.set_bytes(N, 6);
    encoder.set_bytes(K, 7);

    if (use_matvec) {
        constexpr int outputs_per_simdgroup = 4;
        constexpr int simdgroups_per_threadgroup = 2;
        constexpr int outputs_per_threadgroup = outputs_per_simdgroup * simdgroups_per_threadgroup;
        const int column_tiles = (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
        encoder.dispatch_threadgroups(MTL::Size(M * column_tiles, 1, 1),
                                      MTL::Size(simdgroups_per_threadgroup * 32, 1, 1));
        return;
    }

    if (use_tiled) {
        encoder.dispatch_threadgroups(MTL::Size((K + 31) / 32, (M + 31) / 32, 1),
                                      MTL::Size(128, 1, 1));
        return;
    }

    const size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
    const int rows_per_threadgroup = M <= 16 ? 16 : 32;
    const int columns_per_threadgroup = threadgroup_size / rows_per_threadgroup;
    if (columns_per_threadgroup == 0) {
        throw std::runtime_error("quantized_matmul: invalid vanilla threadgroup shape");
    }
    encoder.dispatch_threadgroups(MTL::Size((M + rows_per_threadgroup - 1) / rows_per_threadgroup,
                                            (K + columns_per_threadgroup - 1) / columns_per_threadgroup, 1),
                                  MTL::Size(rows_per_threadgroup, columns_per_threadgroup, 1));
}

// Week 3, Day 4. The earlier Week 2 checkpoints keep the readable row lookup.
mx::array quantized_embedding(const mx::array &, const mx::array &, const mx::array &, const mx::array &, int, int,
                              mx::StreamOrDevice) {
    checkpoint_todo("quantized_embedding", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_cpu", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext
