// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "tiny_llm_ext.h"
#include "axpby.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("load_library", &tiny_llm_ext::load_library, "device"_a, "path"_a);

    m.def("axpby", &tiny_llm_ext::axpby, "x"_a, "y"_a, "alpha"_a, "beta"_a, nb::kw_only(), "stream"_a = nb::none(),
          R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");

    m.def("quantized_matmul", &tiny_llm_ext::quantized_matmul, "scales"_a, "biases"_a, "group_size"_a, "bits"_a,
          "a"_a, "b"_a, "transpose_b"_a = false, "stream"_a = nb::none(),
          R"(
        Matrix multiply with packed int4 weights.

        Args:
            scales (array): Per-group dequantization scales, shape ``(K, N / group_size)``.
            biases (array): Per-group dequantization biases, same shape as ``scales``.
            group_size (int): Number of dense weights per quantization group.
            bits (int): Quantization bit width. This implementation supports 4.
            a (array): Dense activation matrix with shape ``(M, N)``.
            b (array): Packed quantized weights with shape ``(K, N / 8)``.
            transpose_b (bool): Must be true for linear layers, computing ``a @ dequantize(b).T``.

        Returns:
            array: Dense output matrix with shape ``(M, K)``.
      )");
}
