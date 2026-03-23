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
        let clampedN = min(max(1, n), 1024 * 1024)
        let clampedIters = max(1, iters)
        let clampedWarmup = max(0, warmup)

        let expected = cpuExclusiveScanDeterministic(n: clampedN)

        guard let inBuffer = context.device.makeBuffer(length: MemoryLayout<UInt32>.stride * clampedN, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("input buffer")
        }
        guard let outBuffer = context.device.makeBuffer(length: MemoryLayout<UInt32>.stride * clampedN, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("output buffer")
        }

        let inPtr = inBuffer.contents().bindMemory(to: UInt32.self, capacity: clampedN)
        for i in 0..<clampedN { inPtr[i] = UInt32(truncatingIfNeeded: i + 1) }

        var nU32 = UInt32(clampedN)
        guard let nBuffer = context.device.makeBuffer(bytes: &nU32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("n buffer")
        }

        if clampedN <= 1024 {
            let pso = try kernels.pipeline(named: "scan_exclusive_u32_1024")
            let timing = try runSingleGroup(
                context: context,
                pso: pso,
                inBuffer: inBuffer,
                outBuffer: outBuffer,
                nBuffer: nBuffer,
                iters: clampedIters,
                warmup: clampedWarmup,
                n: clampedN
            )
            let ok = verify(outBuffer: outBuffer, expected: expected, n: clampedN)
            return ScanResult(
                n: clampedN,
                iters: clampedIters,
                warmup: clampedWarmup,
                gpuSeconds: timing.gpuSeconds,
                wallSeconds: timing.wallSeconds,
                ok: ok
            )
        } else {
            let psoBlock = try kernels.pipeline(named: "scan_exclusive_u32_block1024")
            let psoSumsScan = try kernels.pipeline(named: "scan_exclusive_u32_1024")
            let psoAddOffsets = try kernels.pipeline(named: "scan_add_block_offsets_u32")

            let timing = try runMultiBlock(
                context: context,
                psoBlock: psoBlock,
                psoSumsScan: psoSumsScan,
                psoAddOffsets: psoAddOffsets,
                inBuffer: inBuffer,
                outBuffer: outBuffer,
                nBuffer: nBuffer,
                iters: clampedIters,
                warmup: clampedWarmup,
                n: clampedN
            )
            let ok = verify(outBuffer: outBuffer, expected: expected, n: clampedN)
            return ScanResult(
                n: clampedN,
                iters: clampedIters,
                warmup: clampedWarmup,
                gpuSeconds: timing.gpuSeconds,
                wallSeconds: timing.wallSeconds,
                ok: ok
            )
        }
    }

    private struct Timing {
        let gpuSeconds: Double
        let wallSeconds: Double
    }

    private static func cpuExclusiveScanDeterministic(n: Int) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: n)
        var running: UInt32 = 0
        for i in 0..<n {
            out[i] = running
            running &+= UInt32(truncatingIfNeeded: i + 1)
        }
        return out
    }

    private static func runSingleGroup(
        context: MetalContext,
        pso: MTLComputePipelineState,
        inBuffer: MTLBuffer,
        outBuffer: MTLBuffer,
        nBuffer: MTLBuffer,
        iters: Int,
        warmup: Int,
        n: Int
    ) throws -> Timing {
        func dispatchOnce() throws {
            guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { throw ScanBenchmarkError.commandBufferFailed }
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw ScanBenchmarkError.computeEncoderFailed }
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(inBuffer, offset: 0, index: 0)
            encoder.setBuffer(outBuffer, offset: 0, index: 1)
            encoder.setBuffer(nBuffer, offset: 0, index: 2)
            encoder.dispatchThreads(MTLSize(width: 512, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        for _ in 0..<warmup { try dispatchOnce() }

        guard let timed = context.commandQueue.makeCommandBuffer() else { throw ScanBenchmarkError.commandBufferFailed }
        timed.label = "scan"
        for _ in 0..<iters {
            guard let encoder = timed.makeComputeCommandEncoder() else { throw ScanBenchmarkError.computeEncoderFailed }
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(inBuffer, offset: 0, index: 0)
            encoder.setBuffer(outBuffer, offset: 0, index: 1)
            encoder.setBuffer(nBuffer, offset: 0, index: 2)
            encoder.dispatchThreads(MTLSize(width: 512, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            encoder.endEncoding()
        }

        let wall0 = DispatchTime.now().uptimeNanoseconds
        timed.commit()
        timed.waitUntilCompleted()
        let wall1 = DispatchTime.now().uptimeNanoseconds
        return Timing(
            gpuSeconds: max(0.0, timed.gpuEndTime - timed.gpuStartTime),
            wallSeconds: Double(wall1 - wall0) / 1_000_000_000.0
        )
    }

    private static func runMultiBlock(
        context: MetalContext,
        psoBlock: MTLComputePipelineState,
        psoSumsScan: MTLComputePipelineState,
        psoAddOffsets: MTLComputePipelineState,
        inBuffer: MTLBuffer,
        outBuffer: MTLBuffer,
        nBuffer: MTLBuffer,
        iters: Int,
        warmup: Int,
        n: Int
    ) throws -> Timing {
        let blockCount = (n + 1023) / 1024
        guard blockCount <= 1024 else {
            throw ScanBenchmarkError.allocationFailed("n too large (max 1024*1024)")
        }

        guard let blockSums = context.device.makeBuffer(length: MemoryLayout<UInt32>.stride * blockCount, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("block sums buffer")
        }
        guard let blockOffsets = context.device.makeBuffer(length: MemoryLayout<UInt32>.stride * blockCount, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("block offsets buffer")
        }

        var blocksU32 = UInt32(blockCount)
        guard let blocksBuffer = context.device.makeBuffer(bytes: &blocksU32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]) else {
            throw ScanBenchmarkError.allocationFailed("blocks buffer")
        }

        func encodeScanBlockPass(into commandBuffer: MTLCommandBuffer) throws {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw ScanBenchmarkError.computeEncoderFailed }
            encoder.setComputePipelineState(psoBlock)
            encoder.setBuffer(inBuffer, offset: 0, index: 0)
            encoder.setBuffer(outBuffer, offset: 0, index: 1)
            encoder.setBuffer(blockSums, offset: 0, index: 2)
            encoder.setBuffer(nBuffer, offset: 0, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: blockCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            encoder.endEncoding()
        }

        func encodeScanBlockSums(into commandBuffer: MTLCommandBuffer) throws {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw ScanBenchmarkError.computeEncoderFailed }
            encoder.setComputePipelineState(psoSumsScan)
            encoder.setBuffer(blockSums, offset: 0, index: 0)
            encoder.setBuffer(blockOffsets, offset: 0, index: 1)
            encoder.setBuffer(blocksBuffer, offset: 0, index: 2)
            encoder.dispatchThreads(MTLSize(width: 512, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            encoder.endEncoding()
        }

        func encodeAddOffsets(into commandBuffer: MTLCommandBuffer) throws {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw ScanBenchmarkError.computeEncoderFailed }
            encoder.setComputePipelineState(psoAddOffsets)
            encoder.setBuffer(outBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockOffsets, offset: 0, index: 1)
            encoder.setBuffer(nBuffer, offset: 0, index: 2)
            let threadsPerThreadgroup = MTLSize(width: min(256, psoAddOffsets.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        func dispatchOnce() throws -> Timing {
            guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { throw ScanBenchmarkError.commandBufferFailed }
            commandBuffer.label = "scan"

            try encodeScanBlockPass(into: commandBuffer)
            try encodeScanBlockSums(into: commandBuffer)
            try encodeAddOffsets(into: commandBuffer)

            let wall0 = DispatchTime.now().uptimeNanoseconds
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let wall1 = DispatchTime.now().uptimeNanoseconds

            return Timing(
                gpuSeconds: max(0.0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime),
                wallSeconds: Double(wall1 - wall0) / 1_000_000_000.0
            )
        }

        for _ in 0..<warmup { _ = try dispatchOnce() }

        var gpuTotal: Double = 0
        var wallTotal: Double = 0
        for _ in 0..<iters {
            let t = try dispatchOnce()
            gpuTotal += t.gpuSeconds
            wallTotal += t.wallSeconds
        }

        return Timing(gpuSeconds: gpuTotal, wallSeconds: wallTotal)
    }

    private static func verify(outBuffer: MTLBuffer, expected: [UInt32], n: Int) -> Bool {
        let ptr = outBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            if ptr[i] != expected[i] { return false }
        }
        return true
    }
}
