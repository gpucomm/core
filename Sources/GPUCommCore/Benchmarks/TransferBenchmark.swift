import Foundation
import Metal

public enum TransferDirection: String, Sendable {
    case h2d
    case d2h
}

public enum TransferStrategy: String, Sendable {
    case memcpy
    case blit
}

public struct TransferResult: Sendable {
    public let direction: TransferDirection
    public let mode: StorageMode
    public let strategy: TransferStrategy
    public let sizeBytes: Int
    public let iters: Int
    public let warmup: Int
    public let wallSeconds: Double
    public let gpuSeconds: Double?
    public let bytesPerSecond: Double

    public var avgMicros: Double {
        guard iters > 0 else { return 0 }
        return (wallSeconds / Double(iters)) * 1_000_000.0
    }

    public var prettyLine: String {
        let kib = Double(sizeBytes) / 1024.0
        let gibps = bytesPerSecond / (1024.0 * 1024.0 * 1024.0)
        if let gpuSeconds {
            return String(
                format: "transfer dir=%@ mode=%@ strategy=%@ size=%.1fKiB iters=%d wall=%.3fms avg=%.3fus gpu=%.3fms throughput=%.2fGiB/s",
                direction.rawValue,
                mode.rawValue,
                strategy.rawValue,
                kib,
                iters,
                wallSeconds * 1000.0,
                avgMicros,
                gpuSeconds * 1000.0,
                gibps
            )
        }
        return String(
            format: "transfer dir=%@ mode=%@ strategy=%@ size=%.1fKiB iters=%d wall=%.3fms avg=%.3fus throughput=%.2fGiB/s",
            direction.rawValue,
            mode.rawValue,
            strategy.rawValue,
            kib,
            iters,
            wallSeconds * 1000.0,
            avgMicros,
            gibps
        )
    }

    public func jsonLine() throws -> String {
        var obj: [String: Any] = [
            "bench": "transfer",
            "direction": direction.rawValue,
            "mode": mode.rawValue,
            "strategy": strategy.rawValue,
            "size_bytes": sizeBytes,
            "iters": iters,
            "warmup": warmup,
            "wall_seconds": wallSeconds,
            "avg_us": avgMicros,
            "bytes_per_second": bytesPerSecond,
        ]
        if let gpuSeconds {
            obj["gpu_seconds"] = gpuSeconds
        }
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}

public enum TransferBenchmarkError: Error, CustomStringConvertible {
    case invalidCombination(mode: StorageMode, strategy: TransferStrategy)
    case allocationFailed(String)
    case commandBufferFailed
    case blitEncoderFailed

    public var description: String {
        switch self {
        case .invalidCombination(let mode, let strategy):
            return "invalid combination: mode=\(mode.rawValue) strategy=\(strategy.rawValue) (use shared+memcpy or private+blit)"
        case .allocationFailed(let what):
            return "allocation failed: \(what)"
        case .commandBufferFailed:
            return "failed to create command buffer"
        case .blitEncoderFailed:
            return "failed to create blit encoder"
        }
    }
}

public enum TransferBenchmark {
    public static func run(
        context: MetalContext,
        sizeBytes: Int,
        iters: Int,
        warmup: Int,
        direction: TransferDirection,
        mode: StorageMode,
        strategy: TransferStrategy
    ) throws -> TransferResult {
        if mode == .shared, strategy != .memcpy {
            throw TransferBenchmarkError.invalidCombination(mode: mode, strategy: strategy)
        }
        if mode == .private, strategy != .blit {
            throw TransferBenchmarkError.invalidCombination(mode: mode, strategy: strategy)
        }

        let clampedIters = max(1, iters)
        let clampedWarmup = max(0, warmup)
        let alignedSize = max(1, sizeBytes)

        var hostSrc = [UInt8](repeating: 0, count: alignedSize)
        for i in 0..<hostSrc.count { hostSrc[i] = UInt8(truncatingIfNeeded: i &* 1315423911) }
        var hostDst = [UInt8](repeating: 0, count: alignedSize)
        var gpuSecondsTotal: Double = 0
        var hasGpuTiming = false

        switch mode {
        case .shared:
            guard let buffer = context.device.makeBuffer(length: alignedSize, options: [.storageModeShared]) else {
                throw TransferBenchmarkError.allocationFailed("shared buffer")
            }

            // Seed for d2h.
            if direction == .d2h {
                _ = hostSrc.withUnsafeBytes { srcBytes in
                    memcpy(buffer.contents(), srcBytes.baseAddress!, alignedSize)
                }
            }

            func oneIter() {
                switch direction {
                case .h2d:
                    _ = hostSrc.withUnsafeBytes { srcBytes in
                        memcpy(buffer.contents(), srcBytes.baseAddress!, alignedSize)
                    }
                case .d2h:
                    _ = hostDst.withUnsafeMutableBytes { dstBytes in
                        memcpy(dstBytes.baseAddress!, buffer.contents(), alignedSize)
                    }
                }
            }

            for _ in 0..<clampedWarmup { oneIter() }
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<clampedIters { oneIter() }
            let t1 = DispatchTime.now().uptimeNanoseconds

            // Prevent optimizing away reads.
            _ = hostDst.first

            let wallSeconds = Double(t1 - t0) / 1_000_000_000.0
            let bytesPerSecond = (Double(alignedSize) * Double(clampedIters)) / max(1e-12, wallSeconds)
            return TransferResult(
                direction: direction,
                mode: mode,
                strategy: strategy,
                sizeBytes: alignedSize,
                iters: clampedIters,
                warmup: clampedWarmup,
                wallSeconds: wallSeconds,
                gpuSeconds: nil,
                bytesPerSecond: bytesPerSecond
            )

        case .private:
            guard
                let staging = context.device.makeBuffer(length: alignedSize, options: [.storageModeShared]),
                let deviceBuffer = context.device.makeBuffer(length: alignedSize, options: [.storageModePrivate])
            else {
                throw TransferBenchmarkError.allocationFailed("staging/private buffers")
            }

            // Seed device buffer for d2h.
            if direction == .d2h {
                _ = hostSrc.withUnsafeBytes { srcBytes in
                    memcpy(staging.contents(), srcBytes.baseAddress!, alignedSize)
                }
                guard let cb = context.commandQueue.makeCommandBuffer() else { throw TransferBenchmarkError.commandBufferFailed }
                guard let blit = cb.makeBlitCommandEncoder() else { throw TransferBenchmarkError.blitEncoderFailed }
                blit.copy(from: staging, sourceOffset: 0, to: deviceBuffer, destinationOffset: 0, size: alignedSize)
                blit.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }

            func oneIter() throws -> Double? {
                if direction == .h2d {
                    _ = hostSrc.withUnsafeBytes { srcBytes in
                        memcpy(staging.contents(), srcBytes.baseAddress!, alignedSize)
                    }
                }

                guard let cb = context.commandQueue.makeCommandBuffer() else { throw TransferBenchmarkError.commandBufferFailed }
                cb.label = "transfer"
                guard let blit = cb.makeBlitCommandEncoder() else { throw TransferBenchmarkError.blitEncoderFailed }

                switch direction {
                case .h2d:
                    blit.copy(from: staging, sourceOffset: 0, to: deviceBuffer, destinationOffset: 0, size: alignedSize)
                case .d2h:
                    blit.copy(from: deviceBuffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: alignedSize)
                }
                blit.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()

                if direction == .d2h {
                    _ = hostDst.withUnsafeMutableBytes { dstBytes in
                        memcpy(dstBytes.baseAddress!, staging.contents(), alignedSize)
                    }
                }
                let gpu = max(0.0, cb.gpuEndTime - cb.gpuStartTime)
                return gpu.isFinite ? gpu : nil
            }

            for _ in 0..<clampedWarmup { _ = try oneIter() }

            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<clampedIters {
                if let g = try oneIter() {
                    gpuSecondsTotal += g
                    hasGpuTiming = true
                }
            }
            let t1 = DispatchTime.now().uptimeNanoseconds

            _ = hostDst.first

            let wallSeconds = Double(t1 - t0) / 1_000_000_000.0
            let bytesPerSecond = (Double(alignedSize) * Double(clampedIters)) / max(1e-12, wallSeconds)
            let gpuSeconds = hasGpuTiming ? gpuSecondsTotal : nil
            return TransferResult(
                direction: direction,
                mode: mode,
                strategy: strategy,
                sizeBytes: alignedSize,
                iters: clampedIters,
                warmup: clampedWarmup,
                wallSeconds: wallSeconds,
                gpuSeconds: gpuSeconds,
                bytesPerSecond: bytesPerSecond
            )
        }
    }
}
