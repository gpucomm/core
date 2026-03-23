import Foundation
import Metal

public struct BandwidthResult: Sendable {
    public let mode: StorageMode
    public let sizeBytes: Int
    public let iters: Int
    public let gpuSeconds: Double
    public let gibPerSecond: Double

    public var prettyLine: String {
        let mib = Double(sizeBytes) / (1024.0 * 1024.0)
        return String(
            format: "bandwidth mode=%@ size=%.1fMiB iters=%d gpu=%.3fms throughput=%.2fGiB/s",
            "\(mode)",
            mib,
            iters,
            gpuSeconds * 1000.0,
            gibPerSecond
        )
    }
}

public enum BandwidthBenchmark {
    public static func run(
        context: MetalContext,
        kernels: KernelLibrary,
        sizeBytes: Int,
        iters: Int,
        mode: StorageMode
    ) throws -> BandwidthResult {
        let pso = try kernels.pipeline(named: "bandwidth_copy")

        let alignedSize = max(4, (sizeBytes / 4) * 4)
        let elementCount = alignedSize / MemoryLayout<UInt32>.size

        let stagingInOptions: MTLResourceOptions = [.storageModeShared, .cpuCacheModeWriteCombined]
        let stagingOutOptions: MTLResourceOptions = [.storageModeShared]

        let stagingIn = context.device.makeBuffer(length: alignedSize, options: stagingInOptions)
        let stagingOut = context.device.makeBuffer(length: alignedSize, options: stagingOutOptions)
        guard let stagingIn, let stagingOut else {
            throw NSError(domain: "gpucomm.core", code: 2, userInfo: [NSLocalizedDescriptionKey: "failed to allocate staging buffers"])
        }

        // Deterministic input pattern (keeps the compiler honest).
        let inPtr = stagingIn.contents().bindMemory(to: UInt32.self, capacity: elementCount)
        for i in 0..<elementCount { inPtr[i] = UInt32(truncatingIfNeeded: i &* 2654435761) }

        let inBuffer: MTLBuffer
        let outBuffer: MTLBuffer

        switch mode {
        case .shared:
            inBuffer = stagingIn
            outBuffer = stagingOut

        case .private:
            guard
                let privIn = context.device.makeBuffer(length: alignedSize, options: [.storageModePrivate]),
                let privOut = context.device.makeBuffer(length: alignedSize, options: [.storageModePrivate])
            else {
                throw NSError(domain: "gpucomm.core", code: 3, userInfo: [NSLocalizedDescriptionKey: "failed to allocate private buffers"])
            }
            inBuffer = privIn
            outBuffer = privOut
        }

        var itersU32 = UInt32(max(1, iters))
        guard let itersBuffer = context.device.makeBuffer(bytes: &itersU32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]) else {
            throw NSError(domain: "gpucomm.core", code: 4, userInfo: [NSLocalizedDescriptionKey: "failed to allocate iters buffer"])
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw NSError(domain: "gpucomm.core", code: 5, userInfo: [NSLocalizedDescriptionKey: "failed to create command buffer"])
        }
        commandBuffer.label = "bandwidth"

        if mode == .private {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw NSError(domain: "gpucomm.core", code: 6, userInfo: [NSLocalizedDescriptionKey: "failed to create blit encoder"])
            }
            blit.copy(from: stagingIn, sourceOffset: 0, to: inBuffer, destinationOffset: 0, size: alignedSize)
            blit.endEncoding()
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NSError(domain: "gpucomm.core", code: 7, userInfo: [NSLocalizedDescriptionKey: "failed to create compute encoder"])
        }
        encoder.label = "bandwidth_copy"
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        encoder.setBuffer(itersBuffer, offset: 0, index: 2)

        let threadsPerThreadgroup = MTLSize(width: min(256, pso.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadsPerGrid = MTLSize(width: elementCount, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        if mode == .private {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw NSError(domain: "gpucomm.core", code: 8, userInfo: [NSLocalizedDescriptionKey: "failed to create blit encoder (download)"])
            }
            blit.copy(from: outBuffer, sourceOffset: 0, to: stagingOut, destinationOffset: 0, size: alignedSize)
            blit.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let gpuSeconds = max(0.0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)

        // Approx: 1 read + 1 write per iter, per element (UInt32).
        let bytesMoved = Double(elementCount) * Double(MemoryLayout<UInt32>.size) * 2.0 * Double(max(1, iters))
        let gibPerSecond = gpuSeconds > 0 ? (bytesMoved / gpuSeconds) / (1024.0 * 1024.0 * 1024.0) : 0.0

        return BandwidthResult(
            mode: mode,
            sizeBytes: alignedSize,
            iters: max(1, iters),
            gpuSeconds: gpuSeconds,
            gibPerSecond: gibPerSecond
        )
    }
}

