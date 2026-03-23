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

// Single-threadgroup exclusive scan for up to 1024 uint elements.
// Host must dispatch exactly 512 threads in 1 threadgroup.
kernel void scan_exclusive_u32_1024(
    device const uint* inBuffer [[buffer(0)]],
    device uint* outBuffer [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup uint data[1024];

    uint i0 = tid * 2u;
    uint i1 = i0 + 1u;

    data[i0] = (i0 < n) ? inBuffer[i0] : 0u;
    data[i1] = (i1 < n) ? inBuffer[i1] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase.
    for (uint stride = 1u; stride < 1024u; stride <<= 1u) {
        uint index = ((tid + 1u) * stride * 2u) - 1u;
        if (index < 1024u) {
            data[index] += data[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear the last element for exclusive scan.
    if (tid == 0u) {
        data[1023] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep phase.
    for (uint stride = 512u; stride >= 1u; stride >>= 1u) {
        uint index = ((tid + 1u) * stride * 2u) - 1u;
        if (index < 1024u) {
            uint t = data[index - stride];
            data[index - stride] = data[index];
            data[index] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stride == 1u) { break; } // avoid uint underflow in loop condition
    }

    if (i0 < n) { outBuffer[i0] = data[i0]; }
    if (i1 < n) { outBuffer[i1] = data[i1]; }
}

// Multi-threadgroup exclusive scan for uint32, per-block 1024 elements.
// Writes:
// - outBuffer: exclusive scan for each element
// - blockSums: total sum for each 1024-element block (inclusive total)
kernel void scan_exclusive_u32_block1024(
    device const uint* inBuffer [[buffer(0)]],
    device uint* outBuffer [[buffer(1)]],
    device uint* blockSums [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint blockId [[threadgroup_position_in_grid]]
) {
    threadgroup uint data[1024];

    uint base = blockId * 1024u;
    uint i0 = base + tid * 2u;
    uint i1 = i0 + 1u;

    data[tid * 2u] = (i0 < n) ? inBuffer[i0] : 0u;
    data[tid * 2u + 1u] = (i1 < n) ? inBuffer[i1] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep.
    for (uint stride = 1u; stride < 1024u; stride <<= 1u) {
        uint index = ((tid + 1u) * stride * 2u) - 1u;
        if (index < 1024u) {
            data[index] += data[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Capture block total before exclusive conversion.
    if (tid == 0u) {
        blockSums[blockId] = data[1023];
        data[1023] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep.
    for (uint stride = 512u; stride >= 1u; stride >>= 1u) {
        uint index = ((tid + 1u) * stride * 2u) - 1u;
        if (index < 1024u) {
            uint t = data[index - stride];
            data[index - stride] = data[index];
            data[index] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stride == 1u) { break; }
    }

    if (i0 < n) { outBuffer[i0] = data[tid * 2u]; }
    if (i1 < n) { outBuffer[i1] = data[tid * 2u + 1u]; }
}

kernel void scan_add_block_offsets_u32(
    device uint* outBuffer [[buffer(0)]],
    device const uint* blockOffsets [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    uint blockId = gid / 1024u;
    outBuffer[gid] += blockOffsets[blockId];
}
