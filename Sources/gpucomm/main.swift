import Foundation
import GPUCommCore

private func usage(_ exitCode: Int32) -> Never {
    let text = """
    gpucomm — Metal-based GPU communication + compute experiments

    Usage:
      gpucomm bench bandwidth [--size-mib N] [--iters N] [--mode shared|private] [--reps N] [--format human|json|csv]
      gpucomm bench bandwidth-sweep [--sizes-mib CSV] [--iters N] [--mode shared|private] [--reps N] [--format human|jsonl|csv]
      gpucomm bench scan [--n N] [--iters N] [--warmup N] [--reps N] [--format human|json|csv]
      gpucomm bench scan-sweep [--ns CSV] [--iters N] [--warmup N] [--reps N] [--format human|jsonl|csv]
      gpucomm bench latency [--kind empty|kernel] [--iters N] [--warmup N] [--reps N] [--format human|json|csv]
      gpucomm bench matmul [--m N] [--n N] [--k N] [--iters N] [--warmup N] [--reps N] [--variant naive|tiled8|tiled16|tiled32] [--tg-x N] [--tg-y N] [--format human|json|csv]
      gpucomm bench matmul-sweep [--m N] [--n N] [--k N] [--iters N] [--warmup N] [--reps N] [--format human|jsonl|csv]
      gpucomm bench transfer [--size-kib N] [--iters N] [--warmup N] [--reps N] [--direction h2d|d2h] [--mode shared|private] [--strategy memcpy|blit] [--format human|json|csv]
      gpucomm bench transfer-sweep [--sizes-kib CSV] [--iters N] [--warmup N] [--reps N] [--direction h2d|d2h|both] [--mode shared|private|both] [--format human|jsonl|csv]
      gpucomm run reduction [--n N]

    Examples:
      gpucomm bench bandwidth --size-mib 64 --iters 200 --mode shared
      gpucomm bench bandwidth --size-mib 64 --iters 200 --mode private
      gpucomm bench bandwidth-sweep --sizes-mib 1,4,16,64 --iters 200 --mode private --format jsonl
      gpucomm bench scan --n 1024 --iters 200 --warmup 20
      gpucomm bench scan-sweep --ns 1024,4096,65536,1048576 --iters 50 --warmup 10 --format jsonl
      gpucomm bench latency --kind kernel --iters 2000 --warmup 200 --reps 5 --format json
      gpucomm bench matmul --m 256 --n 256 --k 256 --iters 50 --warmup 10 --variant tiled16
      gpucomm bench matmul --m 512 --n 512 --k 512 --iters 20 --warmup 5 --variant naive --tg-x 16 --tg-y 8
      gpucomm bench matmul-sweep --m 512 --n 512 --k 512 --iters 10 --warmup 3
      gpucomm bench transfer --size-kib 4 --iters 10000 --warmup 100 --direction h2d --mode private --strategy blit
      gpucomm bench transfer --size-kib 4 --iters 10000 --warmup 100 --direction d2h --mode private --strategy blit --format json
      gpucomm bench transfer-sweep --sizes-kib 1,4,64 --iters 5000 --warmup 200 --direction both --mode both --format jsonl
      gpucomm run reduction --n 1024
    """
    print(text)
    Foundation.exit(exitCode)
}

private func die(_ message: String, exitCode: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data(("error: \(message)\n").utf8))
    Foundation.exit(exitCode)
}

let argv = Array(CommandLine.arguments.dropFirst())
if argv.isEmpty { usage(0) }

// Allow help without requiring a Metal device (useful for CI).
if let first = argv.first, ["-h", "--help", "help"].contains(first) {
    usage(0)
}
if argv.first == "bench" {
    if argv.count == 1 || ["-h", "--help", "help"].contains(argv[1]) {
        usage(0)
    }
}
if argv.first == "run" {
    if argv.count == 1 || ["-h", "--help", "help"].contains(argv[1]) {
        usage(0)
    }
}

guard let context = MetalContext() else {
    die("Metal device not available (are you on macOS with a supported GPU?)")
}

do {
    let kernels = try KernelLibrary(context: context)

    var reader = ArgReader(argv)
    guard let command = reader.pop() else { usage(0) }

    switch command {
    case "bench":
        guard let bench = reader.pop() else { usage(1) }
        switch bench {
        case "bandwidth":
            let sizeMiB = reader.popInt(for: "--size-mib") ?? 64
            let iters = reader.popInt(for: "--iters") ?? 200
            let modeRaw = reader.popValue(for: "--mode") ?? "shared"
            let reps = reader.popInt(for: "--reps") ?? 1
            guard let mode = StorageMode(rawValue: modeRaw) else {
                die("invalid --mode '\(modeRaw)' (expected shared|private)")
            }
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .json)
            if !reader.isEmpty { usage(1) }

            let clampedReps = max(1, reps)
            if clampedReps == 1 {
                let result = try BandwidthBenchmark.run(
                    context: context,
                    kernels: kernels,
                    sizeBytes: sizeMiB * 1024 * 1024,
                    iters: iters,
                    mode: mode
                )
                switch format {
                case .human:
                    print(result.prettyLine)
                case .json, .jsonl:
                    print(try result.jsonLine())
                case .csv:
                    printCSV(
                        header: ["bench", "mode", "size_bytes", "iters", "gpu_seconds", "gib_per_second"],
                        rows: [[
                            "bandwidth",
                            mode.rawValue,
                            "\(result.sizeBytes)",
                            "\(result.iters)",
                            "\(result.gpuSeconds)",
                            "\(result.gibPerSecond)",
                        ]]
                    )
                }
            } else {
                var gibpsSamples: [Double] = []
                var gpuMsSamples: [Double] = []
                gibpsSamples.reserveCapacity(clampedReps)
                gpuMsSamples.reserveCapacity(clampedReps)
                for _ in 0..<clampedReps {
                    let r = try BandwidthBenchmark.run(
                        context: context,
                        kernels: kernels,
                        sizeBytes: sizeMiB * 1024 * 1024,
                        iters: iters,
                        mode: mode
                    )
                    gibpsSamples.append(r.gibPerSecond)
                    gpuMsSamples.append(r.gpuSeconds * 1000.0)
                }
                let gibps = Stats.summarize(gibpsSamples)
                let gpuMs = Stats.summarize(gpuMsSamples)
                let sizeBytes = sizeMiB * 1024 * 1024

                switch format {
                case .human:
                    let mib = Double(sizeBytes) / (1024.0 * 1024.0)
                    print(String(
                        format: "bandwidth mode=%@ size=%.1fMiB iters=%d reps=%d gibps_p50=%.2f gibps_p95=%.2f gpu_ms_p50=%.3f gpu_ms_p95=%.3f",
                        mode.rawValue,
                        mib,
                        iters,
                        gibps.count,
                        gibps.p50,
                        gibps.p95,
                        gpuMs.p50,
                        gpuMs.p95
                    ))
                case .json, .jsonl:
                    let obj: [String: Any] = [
                        "bench": "bandwidth",
                        "mode": mode.rawValue,
                        "size_bytes": sizeBytes,
                        "iters": iters,
                        "reps": gibps.count,
                        "gib_per_second_p50": gibps.p50,
                        "gib_per_second_p95": gibps.p95,
                        "gpu_ms_p50": gpuMs.p50,
                        "gpu_ms_p95": gpuMs.p95,
                    ]
                    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                case .csv:
                    printCSV(
                        header: ["bench", "mode", "size_bytes", "iters", "reps", "gibps_p50", "gibps_p95", "gpu_ms_p50", "gpu_ms_p95"],
                        rows: [[
                            "bandwidth",
                            mode.rawValue,
                            "\(sizeBytes)",
                            "\(iters)",
                            "\(gibps.count)",
                            "\(gibps.p50)",
                            "\(gibps.p95)",
                            "\(gpuMs.p50)",
                            "\(gpuMs.p95)",
                        ]]
                    )
                }
            }

        case "bandwidth-sweep":
            let sizesCSV = reader.popValue(for: "--sizes-mib") ?? "1,4,16,64"
            let iters = reader.popInt(for: "--iters") ?? 200
            let modeRaw = reader.popValue(for: "--mode") ?? "private"
            let reps = reader.popInt(for: "--reps") ?? 1
            guard let mode = StorageMode(rawValue: modeRaw) else {
                die("invalid --mode '\(modeRaw)' (expected shared|private)")
            }
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .jsonl)
            if !reader.isEmpty { usage(1) }

            let sizesMiB = sizesCSV
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .compactMap(Int.init)
                .filter { $0 > 0 }
            if sizesMiB.isEmpty { die("invalid --sizes-mib '\(sizesCSV)' (expected comma-separated ints)") }

            let clampedReps = max(1, reps)
            struct BWPoint {
                let sizeBytes: Int
                let iters: Int
                let mode: StorageMode
                let gibps: StatsSummary
                let gpuMs: StatsSummary
            }

            let points: [BWPoint] = try sizesMiB.map { sizeMiB in
                var gibpsSamples: [Double] = []
                var gpuMsSamples: [Double] = []
                gibpsSamples.reserveCapacity(clampedReps)
                gpuMsSamples.reserveCapacity(clampedReps)

                for _ in 0..<clampedReps {
                    let r = try BandwidthBenchmark.run(
                        context: context,
                        kernels: kernels,
                        sizeBytes: sizeMiB * 1024 * 1024,
                        iters: iters,
                        mode: mode
                    )
                    gibpsSamples.append(r.gibPerSecond)
                    gpuMsSamples.append(r.gpuSeconds * 1000.0)
                }

                return BWPoint(
                    sizeBytes: sizeMiB * 1024 * 1024,
                    iters: iters,
                    mode: mode,
                    gibps: Stats.summarize(gibpsSamples),
                    gpuMs: Stats.summarize(gpuMsSamples)
                )
            }

            switch format {
            case .human:
                for p in points {
                    let mib = Double(p.sizeBytes) / (1024.0 * 1024.0)
                    print(String(
                        format: "bandwidth mode=%@ size=%.1fMiB iters=%d reps=%d gibps_p50=%.2f gibps_p95=%.2f gpu_ms_p50=%.3f gpu_ms_p95=%.3f",
                        p.mode.rawValue,
                        mib,
                        p.iters,
                        p.gibps.count,
                        p.gibps.p50,
                        p.gibps.p95,
                        p.gpuMs.p50,
                        p.gpuMs.p95
                    ))
                }
            case .json, .jsonl:
                for p in points {
                    let obj: [String: Any] = [
                        "bench": "bandwidth",
                        "mode": p.mode.rawValue,
                        "size_bytes": p.sizeBytes,
                        "iters": p.iters,
                        "reps": p.gibps.count,
                        "gib_per_second_p50": p.gibps.p50,
                        "gib_per_second_p95": p.gibps.p95,
                        "gpu_ms_p50": p.gpuMs.p50,
                        "gpu_ms_p95": p.gpuMs.p95,
                    ]
                    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                }
            case .csv:
                printCSV(
                    header: ["bench", "mode", "size_bytes", "iters", "reps", "gibps_p50", "gibps_p95", "gpu_ms_p50", "gpu_ms_p95"],
                    rows: points.map { p in
                        [
                            "bandwidth",
                            p.mode.rawValue,
                            "\(p.sizeBytes)",
                            "\(p.iters)",
                            "\(p.gibps.count)",
                            "\(p.gibps.p50)",
                            "\(p.gibps.p95)",
                            "\(p.gpuMs.p50)",
                            "\(p.gpuMs.p95)",
                        ]
                    }
                )
            }

        case "transfer":
            let sizeKiB = reader.popInt(for: "--size-kib") ?? 4
            let iters = reader.popInt(for: "--iters") ?? 10_000
            let warmup = reader.popInt(for: "--warmup") ?? 100
            let reps = reader.popInt(for: "--reps") ?? 1
            let directionRaw = reader.popValue(for: "--direction") ?? "h2d"
            let modeRaw = reader.popValue(for: "--mode") ?? "private"
            guard let direction = TransferDirection(rawValue: directionRaw) else {
                die("invalid --direction '\(directionRaw)' (expected h2d|d2h)")
            }
            guard let mode = StorageMode(rawValue: modeRaw) else {
                die("invalid --mode '\(modeRaw)' (expected shared|private)")
            }
            let defaultStrategy = (mode == .shared) ? "memcpy" : "blit"
            let strategyRaw = reader.popValue(for: "--strategy") ?? defaultStrategy
            guard let strategy = TransferStrategy(rawValue: strategyRaw) else {
                die("invalid --strategy '\(strategyRaw)' (expected memcpy|blit)")
            }
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .json)
            if !reader.isEmpty { usage(1) }

            let clampedReps = max(1, reps)
            if clampedReps == 1 {
                let result = try TransferBenchmark.run(
                    context: context,
                    sizeBytes: sizeKiB * 1024,
                    iters: iters,
                    warmup: warmup,
                    direction: direction,
                    mode: mode,
                    strategy: strategy
                )
                switch format {
                case .human:
                    print(result.prettyLine)
                case .json, .jsonl:
                    print(try result.jsonLine())
                case .csv:
                    printCSV(
                        header: ["bench", "direction", "mode", "strategy", "size_bytes", "iters", "warmup", "wall_seconds", "gpu_seconds", "bytes_per_second", "avg_us"],
                        rows: [[
                            "transfer",
                            direction.rawValue,
                            mode.rawValue,
                            strategy.rawValue,
                            "\(result.sizeBytes)",
                            "\(result.iters)",
                            "\(result.warmup)",
                            "\(result.wallSeconds)",
                            result.gpuSeconds.map { "\($0)" } ?? "",
                            "\(result.bytesPerSecond)",
                            "\(result.avgMicros)",
                        ]]
                    )
                }
            } else {
                var avgSamples: [Double] = []
                var bpsSamples: [Double] = []
                var gpuMsSamples: [Double] = []
                avgSamples.reserveCapacity(clampedReps)
                bpsSamples.reserveCapacity(clampedReps)
                gpuMsSamples.reserveCapacity(clampedReps)

                for _ in 0..<clampedReps {
                    let r = try TransferBenchmark.run(
                        context: context,
                        sizeBytes: sizeKiB * 1024,
                        iters: iters,
                        warmup: warmup,
                        direction: direction,
                        mode: mode,
                        strategy: strategy
                    )
                    avgSamples.append(r.avgMicros)
                    bpsSamples.append(r.bytesPerSecond)
                    if let g = r.gpuSeconds { gpuMsSamples.append(g * 1000.0) }
                }

                let avgUs = Stats.summarize(avgSamples)
                let bps = Stats.summarize(bpsSamples)
                let gpuMs = gpuMsSamples.isEmpty ? nil : Stats.summarize(gpuMsSamples)
                let sizeBytes = sizeKiB * 1024

                switch format {
                case .human:
                    let kib = Double(sizeBytes) / 1024.0
                    print(String(
                        format: "transfer dir=%@ mode=%@ strategy=%@ size=%.1fKiB iters=%d warmup=%d reps=%d avg_us_p50=%.3f avg_us_p95=%.3f bps_p50=%.0f bps_p95=%.0f%@",
                        direction.rawValue,
                        mode.rawValue,
                        strategy.rawValue,
                        kib,
                        iters,
                        warmup,
                        avgUs.count,
                        avgUs.p50,
                        avgUs.p95,
                        bps.p50,
                        bps.p95,
                        gpuMs.map { String(format: " gpu_ms_p50=%.3f gpu_ms_p95=%.3f", $0.p50, $0.p95) } ?? ""
                    ))
                case .json, .jsonl:
                    var obj: [String: Any] = [
                        "bench": "transfer",
                        "direction": direction.rawValue,
                        "mode": mode.rawValue,
                        "strategy": strategy.rawValue,
                        "size_bytes": sizeBytes,
                        "iters": iters,
                        "warmup": warmup,
                        "reps": avgUs.count,
                        "avg_us_p50": avgUs.p50,
                        "avg_us_p95": avgUs.p95,
                        "bytes_per_second_p50": bps.p50,
                        "bytes_per_second_p95": bps.p95,
                    ]
                    if let g = gpuMs {
                        obj["gpu_ms_p50"] = g.p50
                        obj["gpu_ms_p95"] = g.p95
                    }
                    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                case .csv:
                    printCSV(
                        header: ["bench", "direction", "mode", "strategy", "size_bytes", "iters", "warmup", "reps", "avg_us_p50", "avg_us_p95", "bytes_per_second_p50", "bytes_per_second_p95", "gpu_ms_p50", "gpu_ms_p95"],
                        rows: [[
                            "transfer",
                            direction.rawValue,
                            mode.rawValue,
                            strategy.rawValue,
                            "\(sizeBytes)",
                            "\(iters)",
                            "\(warmup)",
                            "\(avgUs.count)",
                            "\(avgUs.p50)",
                            "\(avgUs.p95)",
                            "\(bps.p50)",
                            "\(bps.p95)",
                            gpuMs.map { "\($0.p50)" } ?? "",
                            gpuMs.map { "\($0.p95)" } ?? "",
                        ]]
                    )
                }
            }

        case "transfer-sweep":
            let sizesCSV = reader.popValue(for: "--sizes-kib") ?? "1,4,64"
            let iters = reader.popInt(for: "--iters") ?? 5000
            let warmup = reader.popInt(for: "--warmup") ?? 200
            let reps = reader.popInt(for: "--reps") ?? 1
            let directionRaw = reader.popValue(for: "--direction") ?? "both"
            let modeRaw = reader.popValue(for: "--mode") ?? "both"
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .jsonl)
            if !reader.isEmpty { usage(1) }

            let sizesKiB = sizesCSV
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .compactMap(Int.init)
                .filter { $0 > 0 }
            if sizesKiB.isEmpty { die("invalid --sizes-kib '\(sizesCSV)' (expected comma-separated ints)") }

            let directions: [TransferDirection]
            switch directionRaw {
            case "h2d":
                directions = [.h2d]
            case "d2h":
                directions = [.d2h]
            case "both":
                directions = [.h2d, .d2h]
            default:
                die("invalid --direction '\(directionRaw)' (expected h2d|d2h|both)")
            }

            let modes: [StorageMode]
            switch modeRaw {
            case "shared":
                modes = [.shared]
            case "private":
                modes = [.private]
            case "both":
                modes = [.shared, .private]
            default:
                die("invalid --mode '\(modeRaw)' (expected shared|private|both)")
            }

            let clampedReps = max(1, reps)
            struct XferPoint {
                let direction: TransferDirection
                let mode: StorageMode
                let strategy: TransferStrategy
                let sizeBytes: Int
                let iters: Int
                let warmup: Int
                let avgUs: StatsSummary
                let bytesPerSec: StatsSummary
                let gpuMs: StatsSummary?
            }

            var points: [XferPoint] = []
            points.reserveCapacity(sizesKiB.count * directions.count * modes.count)

            for sizeKiB in sizesKiB {
                for direction in directions {
                    for mode in modes {
                        let strategy: TransferStrategy = (mode == .shared) ? .memcpy : .blit

                        var avgSamples: [Double] = []
                        var bpsSamples: [Double] = []
                        var gpuMsSamples: [Double] = []
                        avgSamples.reserveCapacity(clampedReps)
                        bpsSamples.reserveCapacity(clampedReps)
                        gpuMsSamples.reserveCapacity(clampedReps)

                        for _ in 0..<clampedReps {
                            let r = try TransferBenchmark.run(
                                context: context,
                                sizeBytes: sizeKiB * 1024,
                                iters: iters,
                                warmup: warmup,
                                direction: direction,
                                mode: mode,
                                strategy: strategy
                            )
                            avgSamples.append(r.avgMicros)
                            bpsSamples.append(r.bytesPerSecond)
                            if let g = r.gpuSeconds {
                                gpuMsSamples.append(g * 1000.0)
                            }
                        }

                        let gpuSummary = gpuMsSamples.isEmpty ? nil : Stats.summarize(gpuMsSamples)
                        points.append(XferPoint(
                            direction: direction,
                            mode: mode,
                            strategy: strategy,
                            sizeBytes: sizeKiB * 1024,
                            iters: iters,
                            warmup: warmup,
                            avgUs: Stats.summarize(avgSamples),
                            bytesPerSec: Stats.summarize(bpsSamples),
                            gpuMs: gpuSummary
                        ))
                    }
                }
            }

            switch format {
            case .human:
                for p in points {
                    let kib = Double(p.sizeBytes) / 1024.0
                    print(String(
                        format: "transfer dir=%@ mode=%@ strategy=%@ size=%.1fKiB iters=%d warmup=%d reps=%d avg_us_p50=%.3f avg_us_p95=%.3f bps_p50=%.0f bps_p95=%.0f%@",
                        p.direction.rawValue,
                        p.mode.rawValue,
                        p.strategy.rawValue,
                        kib,
                        p.iters,
                        p.warmup,
                        p.avgUs.count,
                        p.avgUs.p50,
                        p.avgUs.p95,
                        p.bytesPerSec.p50,
                        p.bytesPerSec.p95,
                        p.gpuMs.map { String(format: " gpu_ms_p50=%.3f gpu_ms_p95=%.3f", $0.p50, $0.p95) } ?? ""
                    ))
                }
            case .json, .jsonl:
                for p in points {
                    var obj: [String: Any] = [
                        "bench": "transfer",
                        "direction": p.direction.rawValue,
                        "mode": p.mode.rawValue,
                        "strategy": p.strategy.rawValue,
                        "size_bytes": p.sizeBytes,
                        "iters": p.iters,
                        "warmup": p.warmup,
                        "reps": p.avgUs.count,
                        "avg_us_p50": p.avgUs.p50,
                        "avg_us_p95": p.avgUs.p95,
                        "bytes_per_second_p50": p.bytesPerSec.p50,
                        "bytes_per_second_p95": p.bytesPerSec.p95,
                    ]
                    if let g = p.gpuMs {
                        obj["gpu_ms_p50"] = g.p50
                        obj["gpu_ms_p95"] = g.p95
                    }
                    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                }
            case .csv:
                printCSV(
                    header: ["bench", "direction", "mode", "strategy", "size_bytes", "iters", "warmup", "reps", "avg_us_p50", "avg_us_p95", "bytes_per_second_p50", "bytes_per_second_p95", "gpu_ms_p50", "gpu_ms_p95"],
                    rows: points.map { p in
                        [
                            "transfer",
                            p.direction.rawValue,
                            p.mode.rawValue,
                            p.strategy.rawValue,
                            "\(p.sizeBytes)",
                            "\(p.iters)",
                            "\(p.warmup)",
                            "\(p.avgUs.count)",
                            "\(p.avgUs.p50)",
                            "\(p.avgUs.p95)",
                            "\(p.bytesPerSec.p50)",
                            "\(p.bytesPerSec.p95)",
                            p.gpuMs.map { "\($0.p50)" } ?? "",
                            p.gpuMs.map { "\($0.p95)" } ?? "",
                        ]
                    }
                )
            }

        case "scan":
            let n = reader.popInt(for: "--n") ?? 1024
            let iters = reader.popInt(for: "--iters") ?? 200
            let warmup = reader.popInt(for: "--warmup") ?? 20
            let reps = reader.popInt(for: "--reps") ?? 1
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .json)
            if !reader.isEmpty { usage(1) }

            let clampedReps = max(1, reps)
            if clampedReps == 1 {
                let result = try ScanBenchmark.run(context: context, kernels: kernels, n: n, iters: iters, warmup: warmup)
                switch format {
                case .human:
                    print(result.prettyLine)
                case .json, .jsonl:
                    print(try result.jsonLine())
                case .csv:
                    printCSV(
                        header: ["bench", "n", "iters", "warmup", "gpu_seconds", "wall_seconds", "avg_us", "ok"],
                        rows: [[
                            "scan",
                            "\(result.n)",
                            "\(result.iters)",
                            "\(result.warmup)",
                            "\(result.gpuSeconds)",
                            "\(result.wallSeconds)",
                            "\(result.avgMicros)",
                            result.ok ? "true" : "false",
                        ]]
                    )
                }
            } else {
                var avgSamples: [Double] = []
                var gpuMsSamples: [Double] = []
                avgSamples.reserveCapacity(clampedReps)
                gpuMsSamples.reserveCapacity(clampedReps)
                var okAll = true

                for _ in 0..<clampedReps {
                    let r = try ScanBenchmark.run(context: context, kernels: kernels, n: n, iters: iters, warmup: warmup)
                    avgSamples.append(r.avgMicros)
                    gpuMsSamples.append(r.gpuSeconds * 1000.0)
                    okAll = okAll && r.ok
                }

                let avgUs = Stats.summarize(avgSamples)
                let gpuMs = Stats.summarize(gpuMsSamples)
                let nClamped = min(max(1, n), 1024 * 1024)

                switch format {
                case .human:
                    print(String(
                        format: "scan n=%d iters=%d warmup=%d reps=%d avg_us_p50=%.3f avg_us_p95=%.3f gpu_ms_p50=%.3f gpu_ms_p95=%.3f ok=%@",
                        nClamped,
                        iters,
                        warmup,
                        avgUs.count,
                        avgUs.p50,
                        avgUs.p95,
                        gpuMs.p50,
                        gpuMs.p95,
                        okAll ? "true" : "false"
                    ))
                case .json, .jsonl:
                    let obj: [String: Any] = [
                        "bench": "scan",
                        "n": nClamped,
                        "iters": iters,
                        "warmup": warmup,
                        "reps": avgUs.count,
                        "avg_us_p50": avgUs.p50,
                        "avg_us_p95": avgUs.p95,
                        "gpu_ms_p50": gpuMs.p50,
                        "gpu_ms_p95": gpuMs.p95,
                        "ok": okAll,
                    ]
                    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                case .csv:
                    printCSV(
                        header: ["bench", "n", "iters", "warmup", "reps", "avg_us_p50", "avg_us_p95", "gpu_ms_p50", "gpu_ms_p95", "ok"],
                        rows: [[
                            "scan",
                            "\(nClamped)",
                            "\(iters)",
                            "\(warmup)",
                            "\(avgUs.count)",
                            "\(avgUs.p50)",
                            "\(avgUs.p95)",
                            "\(gpuMs.p50)",
                            "\(gpuMs.p95)",
                            okAll ? "true" : "false",
                        ]]
                    )
                }
            }

        case "scan-sweep":
            let nsCSV = reader.popValue(for: "--ns") ?? "1024,4096,65536,1048576"
            let iters = reader.popInt(for: "--iters") ?? 50
            let warmup = reader.popInt(for: "--warmup") ?? 10
            let reps = reader.popInt(for: "--reps") ?? 1
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .jsonl)
            if !reader.isEmpty { usage(1) }

            let ns = nsCSV
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .compactMap(Int.init)

            if ns.isEmpty { die("invalid --ns '\(nsCSV)' (expected comma-separated ints)") }

            let clampedReps = max(1, reps)
            struct ScanPoint {
                let n: Int
                let iters: Int
                let warmup: Int
                let ok: Bool
                let avgUs: StatsSummary
                let gpuMs: StatsSummary
            }

            let points: [ScanPoint] = try ns.map { n in
                var avgSamples: [Double] = []
                var gpuMsSamples: [Double] = []
                avgSamples.reserveCapacity(clampedReps)
                gpuMsSamples.reserveCapacity(clampedReps)
                var okAll = true
                for _ in 0..<clampedReps {
                    let r = try ScanBenchmark.run(context: context, kernels: kernels, n: n, iters: iters, warmup: warmup)
                    avgSamples.append(r.avgMicros)
                    gpuMsSamples.append(r.gpuSeconds * 1000.0)
                    okAll = okAll && r.ok
                }
                return ScanPoint(
                    n: n,
                    iters: iters,
                    warmup: warmup,
                    ok: okAll,
                    avgUs: Stats.summarize(avgSamples),
                    gpuMs: Stats.summarize(gpuMsSamples)
                )
            }

            switch format {
            case .human:
                for p in points {
                    print(String(
                        format: "scan n=%d iters=%d warmup=%d reps=%d avg_us_p50=%.3f avg_us_p95=%.3f gpu_ms_p50=%.3f gpu_ms_p95=%.3f ok=%@",
                        p.n,
                        p.iters,
                        p.warmup,
                        p.avgUs.count,
                        p.avgUs.p50,
                        p.avgUs.p95,
                        p.gpuMs.p50,
                        p.gpuMs.p95,
                        p.ok ? "true" : "false"
                    ))
                }
            case .json, .jsonl:
                for p in points {
                    let obj: [String: Any] = [
                        "bench": "scan",
                        "n": p.n,
                        "iters": p.iters,
                        "warmup": p.warmup,
                        "reps": p.avgUs.count,
                        "avg_us_p50": p.avgUs.p50,
                        "avg_us_p95": p.avgUs.p95,
                        "gpu_ms_p50": p.gpuMs.p50,
                        "gpu_ms_p95": p.gpuMs.p95,
                        "ok": p.ok,
                    ]
                    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                }
            case .csv:
                printCSV(
                    header: ["bench", "n", "iters", "warmup", "reps", "avg_us_p50", "avg_us_p95", "gpu_ms_p50", "gpu_ms_p95", "ok"],
                    rows: points.map { p in
                        [
                            "scan",
                            "\(p.n)",
                            "\(p.iters)",
                            "\(p.warmup)",
                            "\(p.avgUs.count)",
                            "\(p.avgUs.p50)",
                            "\(p.avgUs.p95)",
                            "\(p.gpuMs.p50)",
                            "\(p.gpuMs.p95)",
                            p.ok ? "true" : "false",
                        ]
                    }
                )
            }

        case "latency":
            let kindRaw = reader.popValue(for: "--kind") ?? "kernel"
            guard let kind = LatencyKind(rawValue: kindRaw) else {
                die("invalid --kind '\(kindRaw)' (expected empty|kernel)")
            }
            let iters = reader.popInt(for: "--iters") ?? 2000
            let warmup = reader.popInt(for: "--warmup") ?? 200
            let reps = reader.popInt(for: "--reps") ?? 1
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .json)
            if !reader.isEmpty { usage(1) }

            let clampedReps = max(1, reps)
            var wallAvgSamples: [Double] = []
            var gpuAvgSamples: [Double] = []
            wallAvgSamples.reserveCapacity(clampedReps)
            gpuAvgSamples.reserveCapacity(clampedReps)

            for _ in 0..<clampedReps {
                let m = try LatencyBenchmark.runOnce(context: context, kernels: kernels, kind: kind, iters: iters, warmup: warmup)
                wallAvgSamples.append(m.wallAvgMicros)
                if let g = m.gpuAvgMicros { gpuAvgSamples.append(g) }
            }

            let wall = Stats.summarize(wallAvgSamples)
            let gpu = gpuAvgSamples.isEmpty ? nil : Stats.summarize(gpuAvgSamples)

            switch format {
            case .human:
                var tail = ""
                if let gpu {
                    tail = String(format: " gpu_avg_us_p50=%.3f gpu_avg_us_p95=%.3f", gpu.p50, gpu.p95)
                }
                print(String(
                    format: "latency kind=%@ iters=%d warmup=%d reps=%d wall_avg_us_p50=%.3f wall_avg_us_p95=%.3f%@",
                    kind.rawValue,
                    iters,
                    warmup,
                    wall.count,
                    wall.p50,
                    wall.p95,
                    tail
                ))
            case .json, .jsonl:
                var obj: [String: Any] = [
                    "bench": "latency",
                    "kind": kind.rawValue,
                    "iters": iters,
                    "warmup": warmup,
                    "reps": wall.count,
                    "wall_avg_us_p50": wall.p50,
                    "wall_avg_us_p95": wall.p95,
                ]
                if let gpu {
                    obj["gpu_avg_us_p50"] = gpu.p50
                    obj["gpu_avg_us_p95"] = gpu.p95
                }
                let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                print(String(decoding: data, as: UTF8.self))
            case .csv:
                printCSV(
                    header: ["bench", "kind", "iters", "warmup", "reps", "wall_avg_us_p50", "wall_avg_us_p95", "gpu_avg_us_p50", "gpu_avg_us_p95"],
                    rows: [[
                        "latency",
                        kind.rawValue,
                        "\(iters)",
                        "\(warmup)",
                        "\(wall.count)",
                        "\(wall.p50)",
                        "\(wall.p95)",
                        gpu.map { "\($0.p50)" } ?? "",
                        gpu.map { "\($0.p95)" } ?? "",
                    ]]
                )
            }

        case "matmul":
            let m = reader.popInt(for: "--m") ?? 256
            let n = reader.popInt(for: "--n") ?? 256
            let k = reader.popInt(for: "--k") ?? 256
            let iters = reader.popInt(for: "--iters") ?? 50
            let warmup = reader.popInt(for: "--warmup") ?? 10
            let reps = reader.popInt(for: "--reps") ?? 1
            let variantRaw = reader.popValue(for: "--variant") ?? "tiled16"
            guard let variant = MatmulVariant(rawValue: variantRaw) else {
                die("invalid --variant '\(variantRaw)' (expected naive|tiled8|tiled16|tiled32)")
            }
            let tgX = reader.popInt(for: "--tg-x")
            let tgY = reader.popInt(for: "--tg-y")
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .json)
            if !reader.isEmpty { usage(1) }

            let clampedReps = max(1, reps)
            if clampedReps == 1 {
                let result = try MatmulBenchmark.run(
                    context: context,
                    kernels: kernels,
                    m: m,
                    n: n,
                    k: k,
                    iters: iters,
                    warmup: warmup,
                    variant: variant,
                    tgX: tgX,
                    tgY: tgY
                )
                switch format {
                case .human:
                    print(result.prettyLine)
                case .json, .jsonl:
                    print(try result.jsonLine())
                case .csv:
                    printCSV(
                        header: ["bench", "variant", "tg_x", "tg_y", "m", "n", "k", "iters", "warmup", "gpu_seconds", "wall_seconds", "avg_us", "gflops", "ok"],
                        rows: [[
                            "matmul",
                            result.variant.rawValue,
                            result.tgX.map { "\($0)" } ?? "",
                            result.tgY.map { "\($0)" } ?? "",
                            "\(result.m)",
                            "\(result.n)",
                            "\(result.k)",
                            "\(result.iters)",
                            "\(result.warmup)",
                            "\(result.gpuSeconds)",
                            "\(result.wallSeconds)",
                            "\(result.avgMicros)",
                            "\(result.gflops)",
                            result.ok ? "true" : "false",
                        ]]
                    )
                }
            } else {
                var avgSamples: [Double] = []
                var gflopsSamples: [Double] = []
                var gpuMsSamples: [Double] = []
                avgSamples.reserveCapacity(clampedReps)
                gflopsSamples.reserveCapacity(clampedReps)
                gpuMsSamples.reserveCapacity(clampedReps)
                var okAll = true

                var resultMeta: MatmulResult?
                for _ in 0..<clampedReps {
                    let r = try MatmulBenchmark.run(
                        context: context,
                        kernels: kernels,
                        m: m,
                        n: n,
                        k: k,
                        iters: iters,
                        warmup: warmup,
                        variant: variant,
                        tgX: tgX,
                        tgY: tgY
                    )
                    resultMeta = resultMeta ?? r
                    avgSamples.append(r.avgMicros)
                    gflopsSamples.append(r.gflops)
                    gpuMsSamples.append(r.gpuSeconds * 1000.0)
                    okAll = okAll && r.ok
                }

                let avgUs = Stats.summarize(avgSamples)
                let gflops = Stats.summarize(gflopsSamples)
                let gpuMs = Stats.summarize(gpuMsSamples)
                let meta = resultMeta!

                switch format {
                case .human:
                    var head = "matmul var=\(meta.variant.rawValue)"
                    if let x = meta.tgX, let y = meta.tgY { head += " tg=\(x)x\(y)" }
                    print(String(
                        format: "%@ m=%d n=%d k=%d iters=%d warmup=%d reps=%d gflops_p50=%.2f gflops_p95=%.2f avg_us_p50=%.3f avg_us_p95=%.3f gpu_ms_p50=%.3f gpu_ms_p95=%.3f ok=%@",
                        head,
                        meta.m, meta.n, meta.k,
                        meta.iters,
                        meta.warmup,
                        gflops.count,
                        gflops.p50,
                        gflops.p95,
                        avgUs.p50,
                        avgUs.p95,
                        gpuMs.p50,
                        gpuMs.p95,
                        okAll ? "true" : "false"
                    ))
                case .json, .jsonl:
                    var obj: [String: Any] = [
                        "bench": "matmul",
                        "variant": meta.variant.rawValue,
                        "m": meta.m,
                        "n": meta.n,
                        "k": meta.k,
                        "iters": meta.iters,
                        "warmup": meta.warmup,
                        "reps": gflops.count,
                        "gflops_p50": gflops.p50,
                        "gflops_p95": gflops.p95,
                        "avg_us_p50": avgUs.p50,
                        "avg_us_p95": avgUs.p95,
                        "gpu_ms_p50": gpuMs.p50,
                        "gpu_ms_p95": gpuMs.p95,
                        "ok": okAll,
                    ]
                    if let x = meta.tgX, let y = meta.tgY {
                        obj["tg_x"] = x
                        obj["tg_y"] = y
                    }
                    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                case .csv:
                    printCSV(
                        header: ["bench", "variant", "tg_x", "tg_y", "m", "n", "k", "iters", "warmup", "reps", "gflops_p50", "gflops_p95", "avg_us_p50", "avg_us_p95", "gpu_ms_p50", "gpu_ms_p95", "ok"],
                        rows: [[
                            "matmul",
                            meta.variant.rawValue,
                            meta.tgX.map { "\($0)" } ?? "",
                            meta.tgY.map { "\($0)" } ?? "",
                            "\(meta.m)",
                            "\(meta.n)",
                            "\(meta.k)",
                            "\(meta.iters)",
                            "\(meta.warmup)",
                            "\(gflops.count)",
                            "\(gflops.p50)",
                            "\(gflops.p95)",
                            "\(avgUs.p50)",
                            "\(avgUs.p95)",
                            "\(gpuMs.p50)",
                            "\(gpuMs.p95)",
                            okAll ? "true" : "false",
                        ]]
                    )
                }
            }

        case "matmul-sweep":
            let m = reader.popInt(for: "--m") ?? 512
            let n = reader.popInt(for: "--n") ?? 512
            let k = reader.popInt(for: "--k") ?? 512
            let iters = reader.popInt(for: "--iters") ?? 10
            let warmup = reader.popInt(for: "--warmup") ?? 3
            let reps = reader.popInt(for: "--reps") ?? 1
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .jsonl)
            if !reader.isEmpty { usage(1) }

            let candidates: [(Int, Int)] = [
                (8, 8),
                (16, 8),
                (16, 16),
                (32, 8),
            ]

            let clampedReps = max(1, reps)

            struct MatmulPoint {
                let variant: MatmulVariant
                let tgX: Int?
                let tgY: Int?
                let m: Int
                let n: Int
                let k: Int
                let iters: Int
                let warmup: Int
                let ok: Bool
                let avgUs: StatsSummary
                let gflops: StatsSummary
                let gpuMs: StatsSummary
            }

            func runPoint(variant: MatmulVariant, tgX: Int? = nil, tgY: Int? = nil) throws -> MatmulPoint {
                var avgSamples: [Double] = []
                var gflopsSamples: [Double] = []
                var gpuMsSamples: [Double] = []
                avgSamples.reserveCapacity(clampedReps)
                gflopsSamples.reserveCapacity(clampedReps)
                gpuMsSamples.reserveCapacity(clampedReps)
                var okAll = true
                for _ in 0..<clampedReps {
                    let r = try MatmulBenchmark.run(
                        context: context,
                        kernels: kernels,
                        m: m,
                        n: n,
                        k: k,
                        iters: iters,
                        warmup: warmup,
                        variant: variant,
                        tgX: tgX,
                        tgY: tgY
                    )
                    avgSamples.append(r.avgMicros)
                    gflopsSamples.append(r.gflops)
                    gpuMsSamples.append(r.gpuSeconds * 1000.0)
                    okAll = okAll && r.ok
                }
                return MatmulPoint(
                    variant: variant,
                    tgX: tgX,
                    tgY: tgY,
                    m: m,
                    n: n,
                    k: k,
                    iters: iters,
                    warmup: warmup,
                    ok: okAll,
                    avgUs: Stats.summarize(avgSamples),
                    gflops: Stats.summarize(gflopsSamples),
                    gpuMs: Stats.summarize(gpuMsSamples)
                )
            }

            var points: [MatmulPoint] = []
            points.reserveCapacity(candidates.count + 3)

            for (x, y) in candidates {
                points.append(try runPoint(variant: .naive, tgX: x, tgY: y))
            }

            for variant in [MatmulVariant.tiled8, .tiled16, .tiled32] {
                points.append(try runPoint(variant: variant))
            }

            switch format {
            case .human:
                for p in points {
                    var head = "matmul var=\(p.variant.rawValue)"
                    if let tgX = p.tgX, let tgY = p.tgY { head += " tg=\(tgX)x\(tgY)" }
                    print(String(
                        format: "%@ m=%d n=%d k=%d iters=%d reps=%d gflops_p50=%.2f gflops_p95=%.2f avg_us_p50=%.3f avg_us_p95=%.3f gpu_ms_p50=%.3f gpu_ms_p95=%.3f ok=%@",
                        head,
                        p.m, p.n, p.k,
                        p.iters,
                        p.gflops.count,
                        p.gflops.p50,
                        p.gflops.p95,
                        p.avgUs.p50,
                        p.avgUs.p95,
                        p.gpuMs.p50,
                        p.gpuMs.p95,
                        p.ok ? "true" : "false"
                    ))
                }
            case .jsonl, .json:
                for p in points {
                    var obj: [String: Any] = [
                        "bench": "matmul",
                        "variant": p.variant.rawValue,
                        "m": p.m,
                        "n": p.n,
                        "k": p.k,
                        "iters": p.iters,
                        "warmup": p.warmup,
                        "reps": p.gflops.count,
                        "gflops_p50": p.gflops.p50,
                        "gflops_p95": p.gflops.p95,
                        "avg_us_p50": p.avgUs.p50,
                        "avg_us_p95": p.avgUs.p95,
                        "gpu_ms_p50": p.gpuMs.p50,
                        "gpu_ms_p95": p.gpuMs.p95,
                        "ok": p.ok,
                    ]
                    if let tgX = p.tgX, let tgY = p.tgY {
                        obj["tg_x"] = tgX
                        obj["tg_y"] = tgY
                    }
                    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                }
            case .csv:
                printCSV(
                    header: ["bench", "variant", "tg_x", "tg_y", "m", "n", "k", "iters", "warmup", "reps", "gflops_p50", "gflops_p95", "avg_us_p50", "avg_us_p95", "gpu_ms_p50", "gpu_ms_p95", "ok"],
                    rows: points.map { p in
                        [
                            "matmul",
                            p.variant.rawValue,
                            p.tgX.map { "\($0)" } ?? "",
                            p.tgY.map { "\($0)" } ?? "",
                            "\(p.m)",
                            "\(p.n)",
                            "\(p.k)",
                            "\(p.iters)",
                            "\(p.warmup)",
                            "\(p.gflops.count)",
                            "\(p.gflops.p50)",
                            "\(p.gflops.p95)",
                            "\(p.avgUs.p50)",
                            "\(p.avgUs.p95)",
                            "\(p.gpuMs.p50)",
                            "\(p.gpuMs.p95)",
                            p.ok ? "true" : "false",
                        ]
                    }
                )
            }

        default:
            usage(1)
        }

    case "run":
        guard let run = reader.pop() else { usage(1) }
        switch run {
        case "reduction":
            let n = reader.popInt(for: "--n") ?? 1024
            if !reader.isEmpty { usage(1) }

            let result = try ReductionBenchmark.run(context: context, kernels: kernels, n: n)
            print("reduction sum(n=\(result.n)) = \(result.sum) (expected \(result.expected))")

        default:
            usage(1)
        }

    case "-h", "--help", "help":
        usage(0)

    default:
        usage(1)
    }
} catch {
    die(String(describing: error))
}
