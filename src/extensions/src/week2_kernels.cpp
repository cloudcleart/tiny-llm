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

// rms_norm
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

mx::array rope(const mx::array &x, const mx::array &offsets, int dims, float base, bool traditional,
               mx::StreamOrDevice s) {
    require_float_dtype(x, "rope");
    if (x.ndim() != 4 || offsets.dtype() != mx::int32 || offsets.ndim() != 1 || offsets.shape()[0] != x.shape()[0]) {
        throw std::runtime_error("rope: expected x=[B,L,H,D] and one int32 offset per batch row");
    }
    if (dims <= 0 || dims > x.shape()[3] || dims % 2 != 0) {
        throw std::runtime_error("rope: dims must be positive, even, and no larger than the head dimension");
    }
    return mx::array(x.shape(), x.dtype(), std::make_shared<Week2RoPE>(to_stream(s), dims, base, traditional),
                     {x, offsets});
}

mx::array swiglu(const mx::array &gate, const mx::array &up, mx::StreamOrDevice s) {
    require_float_dtype(gate, "swiglu");
    if (gate.dtype() != up.dtype() || gate.shape() != up.shape()) {
        throw std::runtime_error("swiglu: gate and up must have the same shape and dtype");
    }
    return mx::array(gate.shape(), gate.dtype(), std::make_shared<Week2SwiGLU>(to_stream(s)), {gate, up});
}

void Week2RMSNorm::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("rms_norm: the course extension is GPU-only");
}

#ifdef _METAL_

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    // ==================== Step 1: Extract input and output tensors ====================
    // Retrieve input tensor x (shape: [batch, seq_len, dim])
    const auto &x = inputs[0];
    // Retrieve scaling weight (shape: [dim])
    const auto &weight = inputs[1];
    // Get output tensor reference
    auto &out = outputs[0];
    // Allocate GPU memory for the output tensor
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // ==================== Step 2: Get Metal GPU device and kernel ====================
    // Acquire Metal GPU device handle
    auto &device = mx::metal::device(stream().device);
    // Construct kernel name based on data type and retrieve it from tiny_llm_ext library
    // The kernel name will be: week2_rms_norm_f32, week2_rms_norm_f16, or week2_rms_norm_bf16
    auto kernel = device.get_kernel(
        std::string("week2_rms_norm_") + dtype_suffix(out),
        device.get_library("tiny_llm_ext")
    );

    // ==================== Step 3: Create and configure Metal command encoder ====================
    // Get Metal command encoder to submit GPU tasks
    auto &encoder = mx::metal::get_command_encoder(stream());
    // Set the compute kernel to be executed
    encoder.set_compute_pipeline_state(kernel);
    // Bind input tensor x to kernel parameter position 0
    encoder.set_input_array(x, 0);
    // Bind weight tensor to kernel parameter position 1
    encoder.set_input_array(weight, 1);
    // Bind output tensor to kernel parameter position 2
    encoder.set_output_array(out, 2);

    // ==================== Step 4: Calculate and pass scalar parameters ====================
    // Calculate number of rows: total_elements / last_dimension = batch * seq_len
    const int rows = x.size() / x.shape().back();
    // Get the size of the last dimension (hidden dimension)
    const int dim = x.shape().back();
    // Pass rows parameter to kernel position 3
    encoder.set_bytes(rows, 3);
    // Pass dim parameter to kernel position 4
    encoder.set_bytes(dim, 4);
    // Pass epsilon (for numerical stability) to kernel position 5
    encoder.set_bytes(eps_, 5);

    // ==================== Step 5: Configure threads and shared memory ====================
    // Define threads per threadgroup as 256
    constexpr int threads_per_threadgroup = 256;
    // Calculate number of SIMD groups: 256 / 32 = 8
    constexpr int simdgroups_per_threadgroup = threads_per_threadgroup / 32;
    // Allocate shared memory for threadgroup: 8 SIMD groups * 4 bytes(float) = 32 bytes
    encoder.set_threadgroup_memory_length(simdgroups_per_threadgroup * sizeof(float), 0);

    // ==================== Step 6: Dispatch to GPU for execution ====================
    // Launch computation: 'rows' threadgroups with 256 threads each
    // Each threadgroup processes one row of data in parallel
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
    throw std::runtime_error("rope: the course extension is GPU-only");
}

#ifdef _METAL_

void Week2RoPE::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    // ==================== Step 1: Extract input and output tensors ====================
    // Retrieve input tensor x (shape: [batch, length, heads, head_dim])
    const auto &x = inputs[0];
    // Retrieve position offsets (shape: [batch])
    const auto &offsets = inputs[1];
    // Get output tensor reference
    auto &out = outputs[0];
    // Allocate GPU memory for the output tensor
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // ==================== Step 2: Get Metal GPU device and kernel ====================
    // Acquire Metal GPU device handle
    auto &device = mx::metal::device(stream().device);
    // Construct kernel name based on data type and retrieve it from tiny_llm_ext library
    // The kernel name will be: week2_rope_f32, week2_rope_f16, or week2_rope_bf16
    auto kernel = device.get_kernel(
        std::string("week2_rope_") + dtype_suffix(out),
        device.get_library("tiny_llm_ext")
    );
    
    // ==================== Step 3: Create and configure Metal command encoder ====================
    // Get Metal command encoder to submit GPU tasks
    auto &encoder = mx::metal::get_command_encoder(stream());
    // Set the compute kernel to be executed
    encoder.set_compute_pipeline_state(kernel);
    // Bind input tensor x to kernel parameter position 0
    encoder.set_input_array(x, 0);
    // Bind position offsets to kernel parameter position 1
    encoder.set_input_array(offsets, 1);
    // Bind output tensor to kernel parameter position 2
    encoder.set_output_array(out, 2);

    // ==================== Step 4: Extract tensor dimensions and pass parameters ====================
    // Extract batch size from input tensor shape
    const int batch = x.shape()[0];
    // Extract sequence length from input tensor shape
    const int length = x.shape()[1];
    // Extract number of attention heads from input tensor shape
    const int heads = x.shape()[2];
    // Extract head dimension from input tensor shape
    const int head_dim = x.shape()[3];
    // Copy traditional RoPE flag from member variable
    const int traditional = traditional_;
    
    // Pass batch parameter to kernel position 3
    encoder.set_bytes(batch, 3);
    // Pass length parameter to kernel position 4
    encoder.set_bytes(length, 4);
    // Pass heads parameter to kernel position 5
    encoder.set_bytes(heads, 5);
    // Pass head_dim parameter to kernel position 6
    encoder.set_bytes(head_dim, 6);
    // Pass dims (rotation dimension) parameter to kernel position 7
    encoder.set_bytes(dims_, 7);
    // Pass base (frequency base) parameter to kernel position 8
    encoder.set_bytes(base_, 8);
    // Pass traditional flag to kernel position 9
    encoder.set_bytes(traditional, 9);

    // ==================== Step 5: Calculate total work items ====================
    // Define how many heads each thread processes
    constexpr int heads_per_thread = 4;
    // Calculate number of head blocks: ceiling division of heads by heads_per_thread
    const int head_blocks = (heads + heads_per_thread - 1) / heads_per_thread;
    // Calculate total work items to process
    // Each work item handles one element across batch, length, head_blocks and partial head_dim
    // dims_/2: number of rotation pairs
    // head_dim - dims_: number of non-rotated elements per head
    const int work_items = batch * length * head_blocks * (dims_ / 2 + head_dim - dims_);
    // Limit threads to GPU's maximum threads per threadgroup
    const size_t threads = std::min<size_t>(work_items, kernel->maxTotalThreadsPerThreadgroup());

    // ==================== Step 6: Dispatch to GPU for execution ====================
    // Launch computation with work_items total threads
    // Unlike dispatch_threadgroups, dispatch_threads creates individual threads without grouping
    // This is more flexible for irregular workloads
    encoder.dispatch_threads(MTL::Size(work_items, 1, 1), MTL::Size(threads, 1, 1));
}

#else

void Week2RoPE::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}

#endif

void Week2SwiGLU::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("swiglu: the course extension is GPU-only");
}

#ifdef _METAL_

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &gate = inputs[0];
    const auto &up = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &device = mx::metal::device(stream().device);
    auto kernel = device.get_kernel(
        std::string("week2_swiglu_") + dtype_suffix(out),
        device.get_library("tiny_llm_ext"));

    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(gate, 0);
    encoder.set_input_array(up, 1);
    encoder.set_output_array(out, 2);

    const int size = out.size();
    encoder.set_bytes(size, 3);
    const size_t threads = std::min<size_t>(out.size(), kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(threads, 1, 1));
}

#else

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}

#endif

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
