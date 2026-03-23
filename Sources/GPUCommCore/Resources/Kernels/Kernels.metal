#include <metal_stdlib>
using namespace metal;

// Measures global memory traffic by repeatedly reading/writing a single element.
// Bytes moved (approx): 2 * sizeof(uint) per iteration per element.
kernel void bandwidth_copy(
    device volatile const uint* inBuffer [[buffer(0)]],
    device volatile uint* outBuffer [[buffer(1)]],
    constant uint& iters [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    for (uint i = 0; i < iters; i++) {
        uint v = inBuffer[gid];
        outBuffer[gid] = v + i;
    }
}

// Simple single-threadgroup reduction demo: sums up to 1024 float values into out[0].
// Host must dispatch exactly 512 threads in 1 threadgroup.
kernel void reduce_sum_1024(
    device const float* inBuffer [[buffer(0)]],
    device float* outBuffer [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[512];
    float sum = 0.0f;

    uint i0 = tid;
    uint i1 = tid + 512u;

    if (i0 < n) { sum += inBuffer[i0]; }
    if (i1 < n) { sum += inBuffer[i1]; }

    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 256u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        outBuffer[0] = scratch[0];
    }
}
