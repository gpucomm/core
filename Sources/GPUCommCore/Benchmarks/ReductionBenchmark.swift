import Foundation
import Metal

public struct ReductionResult: Sendable {
    public let n: Int
    public let sum: Float
    public let expected: Float
}

public enum ReductionBenchmark {
    public static func run(context: MetalContext, kernels: KernelLibrary, n: Int) throws -> ReductionResult {
        let pso = try kernels.pipeline(named: "reduce_sum_1024")

        let clampedN = min(max(1, n), 1024)
        let inputCount = 1024

        var input = [Float](repeating: 1.0, count: inputCount)
        let expected = Float(clampedN)

        guard let inBuffer = context.device.makeBuffer(bytes: &input, length: MemoryLayout<Float>.stride * inputCount, options: [.storageModeShared]) else {
            throw NSError(domain: "gpucomm.core", code: 20, userInfo: [NSLocalizedDescriptionKey: "failed to allocate input buffer"])
        }
        guard let outBuffer = context.device.makeBuffer(length: MemoryLayout<Float>.stride, options: [.storageModeShared]) else {
            throw NSError(domain: "gpucomm.core", code: 21, userInfo: [NSLocalizedDescriptionKey: "failed to allocate output buffer"])
        }

        var nU32 = UInt32(clampedN)
        guard let nBuffer = context.device.makeBuffer(bytes: &nU32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]) else {
            throw NSError(domain: "gpucomm.core", code: 22, userInfo: [NSLocalizedDescriptionKey: "failed to allocate n buffer"])
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw NSError(domain: "gpucomm.core", code: 23, userInfo: [NSLocalizedDescriptionKey: "failed to create command buffer"])
        }
        commandBuffer.label = "reduction"

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NSError(domain: "gpucomm.core", code: 24, userInfo: [NSLocalizedDescriptionKey: "failed to create compute encoder"])
        }
        encoder.label = "reduce_sum_1024"
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        encoder.setBuffer(nBuffer, offset: 0, index: 2)

        // Kernel expects exactly 512 threads (reduces up to 1024 elements).
        let threadsPerThreadgroup = MTLSize(width: 512, height: 1, depth: 1)
        let threadsPerGrid = MTLSize(width: 512, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let sum = outBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0]
        return ReductionResult(n: clampedN, sum: sum, expected: expected)
    }
}

