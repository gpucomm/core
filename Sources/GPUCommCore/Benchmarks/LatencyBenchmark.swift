import Foundation
import Metal

public enum LatencyKind: String, Sendable {
    case empty
    case kernel
}

public struct LatencyMeasurement: Sendable {
    public let kind: LatencyKind
    public let iters: Int
    public let warmup: Int
    public let wallSeconds: Double
    public let gpuSeconds: Double?

    public var wallAvgMicros: Double {
        guard iters > 0 else { return 0 }
        return (wallSeconds / Double(iters)) * 1_000_000.0
    }

    public var gpuAvgMicros: Double? {
        guard let gpuSeconds, iters > 0 else { return nil }
        return (gpuSeconds / Double(iters)) * 1_000_000.0
    }
}

public enum LatencyBenchmarkError: Error, CustomStringConvertible {
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

public enum LatencyBenchmark {
    public static func runOnce(
        context: MetalContext,
        kernels: KernelLibrary,
        kind: LatencyKind,
        iters: Int,
        warmup: Int
    ) throws -> LatencyMeasurement {
        let clampedIters = max(1, iters)
        let clampedWarmup = max(0, warmup)

        switch kind {
        case .empty:
            for _ in 0..<clampedWarmup {
                guard let cb = context.commandQueue.makeCommandBuffer() else { throw LatencyBenchmarkError.commandBufferFailed }
                cb.commit()
                cb.waitUntilCompleted()
            }

            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<clampedIters {
                guard let cb = context.commandQueue.makeCommandBuffer() else { throw LatencyBenchmarkError.commandBufferFailed }
                cb.commit()
                cb.waitUntilCompleted()
            }
            let t1 = DispatchTime.now().uptimeNanoseconds

            return LatencyMeasurement(
                kind: kind,
                iters: clampedIters,
                warmup: clampedWarmup,
                wallSeconds: Double(t1 - t0) / 1_000_000_000.0,
                gpuSeconds: nil
            )

        case .kernel:
            let pso = try kernels.pipeline(named: "latency_noop_u32")

            guard let inBuf = context.device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: [.storageModeShared]) else {
                throw LatencyBenchmarkError.allocationFailed("input buffer")
            }
            guard let outBuf = context.device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: [.storageModeShared]) else {
                throw LatencyBenchmarkError.allocationFailed("output buffer")
            }
            inBuf.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = 1

            func one() throws -> Double? {
                guard let cb = context.commandQueue.makeCommandBuffer() else { throw LatencyBenchmarkError.commandBufferFailed }
                cb.label = "latency.kernel"
                guard let enc = cb.makeComputeCommandEncoder() else { throw LatencyBenchmarkError.computeEncoderFailed }
                enc.setComputePipelineState(pso)
                enc.setBuffer(inBuf, offset: 0, index: 0)
                enc.setBuffer(outBuf, offset: 0, index: 1)
                enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                let gpu = max(0.0, cb.gpuEndTime - cb.gpuStartTime)
                return gpu.isFinite ? gpu : nil
            }

            for _ in 0..<clampedWarmup { _ = try one() }

            var gpuTotal: Double = 0
            var gpuCount: Int = 0
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<clampedIters {
                if let g = try one() {
                    gpuTotal += g
                    gpuCount += 1
                }
            }
            let t1 = DispatchTime.now().uptimeNanoseconds

            // Touch output to keep the compiler honest.
            _ = outBuf.contents().bindMemory(to: UInt32.self, capacity: 1)[0]

            return LatencyMeasurement(
                kind: kind,
                iters: clampedIters,
                warmup: clampedWarmup,
                wallSeconds: Double(t1 - t0) / 1_000_000_000.0,
                gpuSeconds: gpuCount > 0 ? gpuTotal : nil
            )
        }
    }
}

