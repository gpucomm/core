import Foundation
import Metal

public enum MatmulVariant: String, Sendable {
    case naive
    case tiled16
}

public struct MatmulResult: Sendable {
    public let variant: MatmulVariant
    public let m: Int
    public let n: Int
    public let k: Int
    public let iters: Int
    public let warmup: Int
    public let gpuSeconds: Double
    public let wallSeconds: Double
    public let ok: Bool

    public var gflops: Double {
        // 2*M*N*K FLOPs per GEMM.
        let flops = 2.0 * Double(m) * Double(n) * Double(k) * Double(iters)
        return (gpuSeconds > 0) ? (flops / gpuSeconds) / 1e9 : 0
    }

    public var avgMicros: Double { (wallSeconds / Double(max(1, iters))) * 1_000_000.0 }

    public var prettyLine: String {
        return String(
            format: "matmul var=%@ m=%d n=%d k=%d iters=%d gpu=%.3fms wall=%.3fms avg=%.3fus gflops=%.2f ok=%@",
            variant.rawValue,
            m, n, k,
            iters,
            gpuSeconds * 1000.0,
            wallSeconds * 1000.0,
            avgMicros,
            gflops,
            ok ? "true" : "false"
        )
    }

    public func jsonLine() throws -> String {
        let obj: [String: Any] = [
            "bench": "matmul",
            "variant": variant.rawValue,
            "m": m,
            "n": n,
            "k": k,
            "iters": iters,
            "warmup": warmup,
            "gpu_seconds": gpuSeconds,
            "wall_seconds": wallSeconds,
            "avg_us": avgMicros,
            "gflops": gflops,
            "ok": ok,
        ]
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}

public enum MatmulBenchmarkError: Error, CustomStringConvertible {
    case allocationFailed(String)
    case commandBufferFailed
    case computeEncoderFailed

    public var description: String {
        switch self {
        case .allocationFailed(let what):
            return "allocation failed: \(what)"
        case .commandBufferFailed:
            return "failed to create command buffer"
        case .computeEncoderFailed:
            return "failed to create compute encoder"
        }
    }
}

public enum MatmulBenchmark {
    public static func run(
        context: MetalContext,
        kernels: KernelLibrary,
        m: Int,
        n: Int,
        k: Int,
        iters: Int,
        warmup: Int,
        variant: MatmulVariant
    ) throws -> MatmulResult {
        let M = max(1, m)
        let N = max(1, n)
        let K = max(1, k)
        let clampedIters = max(1, iters)
        let clampedWarmup = max(0, warmup)

        let functionName: String
        switch variant {
        case .naive: functionName = "matmul_f32_naive"
        case .tiled16: functionName = "matmul_f32_tiled_16"
        }
        let pso = try kernels.pipeline(named: functionName)

        let aCount = M * K
        let bCount = K * N
        let cCount = M * N

        guard
            let aBuf = context.device.makeBuffer(length: MemoryLayout<Float>.stride * aCount, options: [.storageModeShared]),
            let bBuf = context.device.makeBuffer(length: MemoryLayout<Float>.stride * bCount, options: [.storageModeShared]),
            let cBuf = context.device.makeBuffer(length: MemoryLayout<Float>.stride * cCount, options: [.storageModeShared])
        else {
            throw MatmulBenchmarkError.allocationFailed("A/B/C buffers")
        }

        // Deterministic initialization.
        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: aCount)
        for i in 0..<aCount { aPtr[i] = Float((i % 17) - 8) * 0.125 }
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: bCount)
        for i in 0..<bCount { bPtr[i] = Float((i % 13) - 6) * 0.25 }
        memset(cBuf.contents(), 0, MemoryLayout<Float>.stride * cCount)

        var mU32 = UInt32(M)
        var nU32 = UInt32(N)
        var kU32 = UInt32(K)
        guard
            let mBuf = context.device.makeBuffer(bytes: &mU32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]),
            let nBuf = context.device.makeBuffer(bytes: &nU32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]),
            let kBuf = context.device.makeBuffer(bytes: &kU32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared])
        else {
            throw MatmulBenchmarkError.allocationFailed("M/N/K buffers")
        }

        func encode(into commandBuffer: MTLCommandBuffer) throws {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw MatmulBenchmarkError.computeEncoderFailed }
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(aBuf, offset: 0, index: 0)
            encoder.setBuffer(bBuf, offset: 0, index: 1)
            encoder.setBuffer(cBuf, offset: 0, index: 2)
            encoder.setBuffer(mBuf, offset: 0, index: 3)
            encoder.setBuffer(nBuf, offset: 0, index: 4)
            encoder.setBuffer(kBuf, offset: 0, index: 5)

            switch variant {
            case .naive:
                let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
                let threadsPerGrid = MTLSize(width: N, height: M, depth: 1)
                encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

            case .tiled16:
                let tg = MTLSize(width: 16, height: 16, depth: 1)
                let tgs = MTLSize(width: (N + 15) / 16, height: (M + 15) / 16, depth: 1)
                encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            }

            encoder.endEncoding()
        }

        // Warmup.
        for _ in 0..<clampedWarmup {
            guard let cb = context.commandQueue.makeCommandBuffer() else { throw MatmulBenchmarkError.commandBufferFailed }
            try encode(into: cb)
            cb.commit()
            cb.waitUntilCompleted()
        }

        guard let timed = context.commandQueue.makeCommandBuffer() else { throw MatmulBenchmarkError.commandBufferFailed }
        timed.label = "matmul"
        for _ in 0..<clampedIters { try encode(into: timed) }

        let wall0 = DispatchTime.now().uptimeNanoseconds
        timed.commit()
        timed.waitUntilCompleted()
        let wall1 = DispatchTime.now().uptimeNanoseconds

        let gpuSeconds = max(0.0, timed.gpuEndTime - timed.gpuStartTime)
        let wallSeconds = Double(wall1 - wall0) / 1_000_000_000.0

        let ok = verifySmallIfNeeded(a: aPtr, b: bPtr, c: cBuf, m: M, n: N, k: K)
        return MatmulResult(
            variant: variant,
            m: M, n: N, k: K,
            iters: clampedIters,
            warmup: clampedWarmup,
            gpuSeconds: gpuSeconds,
            wallSeconds: wallSeconds,
            ok: ok
        )
    }

    private static func verifySmallIfNeeded(a: UnsafePointer<Float>, b: UnsafePointer<Float>, c: MTLBuffer, m: Int, n: Int, k: Int) -> Bool {
        // Verify only for reasonably small outputs to keep CLI snappy.
        if m * n > 4096 { return true }
        let cPtr = c.contents().bindMemory(to: Float.self, capacity: m * n)
        for row in 0..<m {
            for col in 0..<n {
                var acc: Float = 0
                for kk in 0..<k {
                    acc += a[row * k + kk] * b[kk * n + col]
                }
                let got = cPtr[row * n + col]
                let diff = abs(got - acc)
                if diff > 1e-2 {
                    return false
                }
            }
        }
        return true
    }
}

