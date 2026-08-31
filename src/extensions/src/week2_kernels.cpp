#include <algorithm>
#include <stdexcept>
#include <string>

#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

namespace {

void require_float_dtype(const mx::array &x, const char *name) {
    if (x.dtype() != mx::float32 && x.dtype() != mx::float16 && x.dtype() != mx::bfloat16) {
        throw std::runtime_error(std::string(name) + ": expected float32, float16, or bfloat16");
    }
}

const char *dtype_suffix(const mx::array &x) {
    if (x.dtype() == mx::float32) {
        return "f32";
    }
    if (x.dtype() == mx::float16) {
        return "f16";
    }
    if (x.dtype() == mx::bfloat16) {
        return "bf16";
    }
    throw std::runtime_error("unsupported dtype");
}

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 2, Day 4.
mx::array rms_norm(const mx::array &x, const mx::array &weight, float eps, mx::StreamOrDevice s) {
    require_float_dtype(x, "rms_norm");
    if (x.ndim() == 0) {
        throw std::runtime_error("rms_norm: input must have at least one dimension");
    }
    if (x.dtype() != weight.dtype() || weight.ndim() != 1 || weight.shape()[0] != x.shape().back()) {
        throw std::runtime_error("rms_norm: weight must match the input dtype and final dimension");
    }
    return mx::array(x.shape(), x.dtype(), std::make_shared<Week2RMSNorm>(to_stream(s), eps), {x, weight});
}

mx::array rope(const mx::array &, const mx::array &, int, float, bool, mx::StreamOrDevice) {
    checkpoint_todo("rope", "Week 2, Day 4");
}

mx::array swiglu(const mx::array &, const mx::array &, mx::StreamOrDevice) {
    checkpoint_todo("swiglu", "Week 2, Day 4");
}

void Week2RMSNorm::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("rms_norm: the course extension is GPU-only");
}

#ifdef _METAL_

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &device = mx::metal::device(stream().device);
    auto kernel = device.get_kernel(
        std::string("week2_rms_norm_") + dtype_suffix(out),
        device.get_library("tiny_llm_ext")
    );
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_output_array(out, 2);

    const int rows = x.size() / x.shape().back();
    const int dim = x.shape().back();
    encoder.set_bytes(rows, 3);
    encoder.set_bytes(dim, 4);
    encoder.set_bytes(eps_, 5);

    constexpr int threads_per_threadgroup = 256;
    constexpr int simdgroups_per_threadgroup = threads_per_threadgroup / 32;
    encoder.set_threadgroup_memory_length(simdgroups_per_threadgroup * sizeof(float), 0);
    encoder.dispatch_threadgroups(
        MTL::Size(rows, 1, 1),
        MTL::Size(threads_per_threadgroup, 1, 1)
    );
}

#else

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}

#endif

void Week2RoPE::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2RoPE::eval_cpu", "Week 2, Day 4");
}

void Week2RoPE::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2RoPE::eval_gpu", "Week 2, Day 4");
}

void Week2SwiGLU::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2SwiGLU::eval_cpu", "Week 2, Day 4");
}

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2SwiGLU::eval_gpu", "Week 2, Day 4");
}

// Week 2, Day 5.
mx::array decode_attention(const mx::array &, const mx::array &, const mx::array &, const mx::array &, float, bool,
                           bool, int, int, mx::StreamOrDevice) {
    checkpoint_todo("decode_attention", "Week 2, Day 5");
}

void Week2DecodeAttention::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2DecodeAttention::eval_cpu", "Week 2, Day 5");
}

void Week2DecodeAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2DecodeAttention::eval_gpu", "Week 2, Day 5");
}

}  // namespace tiny_llm_ext
