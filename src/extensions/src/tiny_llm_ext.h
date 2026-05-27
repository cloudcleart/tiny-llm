#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

void load_library(mx::Device d, const char *path);

mx::array quantized_matmul(const mx::array &scales,   // Per-group dequantization scales.
                           const mx::array &biases,   // Per-group dequantization biases.
                           const int group_size,      // Number of original weights per quantization group.
                           const int bits,            // Quantization bit width.
                           const mx::array &a,        // Dense activation matrix, shape (M, N).
                           const mx::array &b,        // Packed quantized weight matrix, shape (K, N / 8).
                           const bool transpose_b,    // Whether to compute against dequantized b^T.
                           mx::StreamOrDevice s = {}  // Stream on which to schedule the operation.
);

class QuantizedMatmul : public mx::Primitive {
public:
    explicit QuantizedMatmul(mx::Stream stream) : mx::Primitive(stream) {};

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override {
        throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
    }

    const char *name() const override { return "QuantizedMatmul"; }
};

}  // namespace tiny_llm_ext
