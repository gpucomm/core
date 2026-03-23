import Foundation
import Metal

public struct ScanResult: Sendable {
    public let n: Int
    public let iters: Int
    public let warmup: Int
    public let gpuSeconds: Double
    public let wallSeconds: Double
    public let ok: Bool

    public var avgMicros: Double { (wallSeconds / Double(max(1, iters))) * 1_000_000.0 }

    public var prettyLine: String {
        return String(
            format: "scan n=%d iters=%d gpu=%.3fms wall=%.3fms avg=%.3fus ok=%@",
            n,
            iters,
            gpuSeconds * 1000.0,
            wallSeconds * 1000.0,
            avgMicros,
            ok ? "true" : "false"
        )
    }

    public func jsonLine() throws -> String {
        let obj: [String: Any] = [
            "bench": "scan",
            "n": n,
            "iters": iters,
            "warmup": warmup,
            "gpu_seconds": gpuSeconds,
            "wall_seconds": wallSeconds,
            "avg_us": avgMicros,
            "ok": ok,
        ]
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}

public enum ScanBenchmarkError: Error, CustomStringConvertible {
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

public enum ScanBenchmark {
    public static func run(
        context: MetalContext,
        kernels: KernelLibrary,
        n: Int,
        iters: Int,
        warmup: Int
    ) throws -> ScanResult {
        let clampedN = min(max(1, n), 1024)
        let clampedIters = max(1, iters)
        let clampedWarmup = max(0, warmup)

        let pso = try kernels.pipeline(named: "scan_exclusive_u32_1024")

        var input = [UInt32](repeating: 0, count: 1024)
        for i in 0..<clampedN { input[i] = UInt32(truncatingIfNeeded: i + 1) }

        let expected = cpuExclusiveScan(input, n: clampedN)

        guard let inBuffer = context.device.makeBuffer(bytes: &input, length: MemoryLayout<UInt32>.stride * 1024, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("input buffer")
        }
        guard let outBuffer = context.device.makeBuffer(length: MemoryLayout<UInt32>.stride * 1024, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("output buffer")
        }

        var nU32 = UInt32(clampedN)
        guard let nBuffer = context.device.makeBuffer(bytes: &nU32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("n buffer")
        }

        func dispatchOnce() throws {
            guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { throw ScanBenchmarkError.commandBufferFailed }
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw ScanBenchmarkError.computeEncoderFailed }
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(inBuffer, offset: 0, index: 0)
            encoder.setBuffer(outBuffer, offset: 0, index: 1)
            encoder.setBuffer(nBuffer, offset: 0, index: 2)

            let threadsPerThreadgroup = MTLSize(width: 512, height: 1, depth: 1)
            let threadsPerGrid = MTLSize(width: 512, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        for _ in 0..<clampedWarmup { try dispatchOnce() }

        guard let timed = context.commandQueue.makeCommandBuffer() else { throw ScanBenchmarkError.commandBufferFailed }
        timed.label = "scan"

        for _ in 0..<clampedIters {
            guard let encoder = timed.makeComputeCommandEncoder() else { throw ScanBenchmarkError.computeEncoderFailed }
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(inBuffer, offset: 0, index: 0)
            encoder.setBuffer(outBuffer, offset: 0, index: 1)
            encoder.setBuffer(nBuffer, offset: 0, index: 2)
            let threadsPerThreadgroup = MTLSize(width: 512, height: 1, depth: 1)
            let threadsPerGrid = MTLSize(width: 512, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        let wall0 = DispatchTime.now().uptimeNanoseconds
        timed.commit()
        timed.waitUntilCompleted()
        let wall1 = DispatchTime.now().uptimeNanoseconds

        let gpuSeconds = max(0.0, timed.gpuEndTime - timed.gpuStartTime)
        let wallSeconds = Double(wall1 - wall0) / 1_000_000_000.0

        let ok = verify(outBuffer: outBuffer, expected: expected, n: clampedN)
        return ScanResult(
            n: clampedN,
            iters: clampedIters,
            warmup: clampedWarmup,
            gpuSeconds: gpuSeconds,
            wallSeconds: wallSeconds,
            ok: ok
        )
    }

    private static func cpuExclusiveScan(_ input: [UInt32], n: Int) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: n)
        var running: UInt32 = 0
        for i in 0..<n {
            out[i] = running
            running &+= input[i]
        }
        return out
    }

    private static func verify(outBuffer: MTLBuffer, expected: [UInt32], n: Int) -> Bool {
        let ptr = outBuffer.contents().bindMemory(to: UInt32.self, capacity: 1024)
        for i in 0..<n {
            if ptr[i] != expected[i] { return false }
        }
        return true
    }
}

