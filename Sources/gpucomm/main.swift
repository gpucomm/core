import Foundation
import GPUCommCore

private func usage(_ exitCode: Int32) -> Never {
    let text = """
    gpucomm — Metal-based GPU communication + compute experiments

    Usage:
      gpucomm bench bandwidth [--size-mib N] [--iters N] [--mode shared|private] [--format human|json|csv]
      gpucomm bench bandwidth-sweep [--sizes-mib CSV] [--iters N] [--mode shared|private] [--format human|jsonl|csv]
      gpucomm bench scan [--n N] [--iters N] [--warmup N] [--format human|json|csv]
      gpucomm bench scan-sweep [--ns CSV] [--iters N] [--warmup N] [--format human|jsonl|csv]
      gpucomm bench matmul [--m N] [--n N] [--k N] [--iters N] [--warmup N] [--variant naive|tiled8|tiled16|tiled32] [--tg-x N] [--tg-y N] [--format human|json|csv]
      gpucomm bench matmul-sweep [--m N] [--n N] [--k N] [--iters N] [--warmup N] [--format human|jsonl|csv]
      gpucomm bench transfer [--size-kib N] [--iters N] [--warmup N] [--direction h2d|d2h] [--mode shared|private] [--strategy memcpy|blit] [--format human|json|csv]
      gpucomm bench transfer-sweep [--sizes-kib CSV] [--iters N] [--warmup N] [--direction h2d|d2h|both] [--mode shared|private|both] [--format human|jsonl|csv]
      gpucomm run reduction [--n N]

    Examples:
      gpucomm bench bandwidth --size-mib 64 --iters 200 --mode shared
      gpucomm bench bandwidth --size-mib 64 --iters 200 --mode private
      gpucomm bench bandwidth-sweep --sizes-mib 1,4,16,64 --iters 200 --mode private --format jsonl
      gpucomm bench scan --n 1024 --iters 200 --warmup 20
      gpucomm bench scan-sweep --ns 1024,4096,65536,1048576 --iters 50 --warmup 10 --format jsonl
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
            guard let mode = StorageMode(rawValue: modeRaw) else {
                die("invalid --mode '\(modeRaw)' (expected shared|private)")
            }
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .json)
            if !reader.isEmpty { usage(1) }

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

        case "bandwidth-sweep":
            let sizesCSV = reader.popValue(for: "--sizes-mib") ?? "1,4,16,64"
            let iters = reader.popInt(for: "--iters") ?? 200
            let modeRaw = reader.popValue(for: "--mode") ?? "private"
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

            let results = try sizesMiB.map { sizeMiB in
                try BandwidthBenchmark.run(
                    context: context,
                    kernels: kernels,
                    sizeBytes: sizeMiB * 1024 * 1024,
                    iters: iters,
                    mode: mode
                )
            }

            switch format {
            case .human:
                for r in results { print(r.prettyLine) }
            case .json, .jsonl:
                for r in results { print(try r.jsonLine()) }
            case .csv:
                printCSV(
                    header: ["bench", "mode", "size_bytes", "iters", "gpu_seconds", "gib_per_second"],
                    rows: results.map { r in
                        [
                            "bandwidth",
                            r.mode.rawValue,
                            "\(r.sizeBytes)",
                            "\(r.iters)",
                            "\(r.gpuSeconds)",
                            "\(r.gibPerSecond)",
                        ]
                    }
                )
            }

        case "transfer":
            let sizeKiB = reader.popInt(for: "--size-kib") ?? 4
            let iters = reader.popInt(for: "--iters") ?? 10_000
            let warmup = reader.popInt(for: "--warmup") ?? 100
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

        case "transfer-sweep":
            let sizesCSV = reader.popValue(for: "--sizes-kib") ?? "1,4,64"
            let iters = reader.popInt(for: "--iters") ?? 5000
            let warmup = reader.popInt(for: "--warmup") ?? 200
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

            var results: [TransferResult] = []
            results.reserveCapacity(sizesKiB.count * directions.count * modes.count)

            for sizeKiB in sizesKiB {
                for direction in directions {
                    for mode in modes {
                        let strategy: TransferStrategy = (mode == .shared) ? .memcpy : .blit
                        results.append(try TransferBenchmark.run(
                            context: context,
                            sizeBytes: sizeKiB * 1024,
                            iters: iters,
                            warmup: warmup,
                            direction: direction,
                            mode: mode,
                            strategy: strategy
                        ))
                    }
                }
            }

            switch format {
            case .human:
                for r in results { print(r.prettyLine) }
            case .json, .jsonl:
                for r in results { print(try r.jsonLine()) }
            case .csv:
                printCSV(
                    header: ["bench", "direction", "mode", "strategy", "size_bytes", "iters", "warmup", "wall_seconds", "gpu_seconds", "bytes_per_second", "avg_us"],
                    rows: results.map { r in
                        [
                            "transfer",
                            r.direction.rawValue,
                            r.mode.rawValue,
                            r.strategy.rawValue,
                            "\(r.sizeBytes)",
                            "\(r.iters)",
                            "\(r.warmup)",
                            "\(r.wallSeconds)",
                            r.gpuSeconds.map { "\($0)" } ?? "",
                            "\(r.bytesPerSecond)",
                            "\(r.avgMicros)",
                        ]
                    }
                )
            }

        case "scan":
            let n = reader.popInt(for: "--n") ?? 1024
            let iters = reader.popInt(for: "--iters") ?? 200
            let warmup = reader.popInt(for: "--warmup") ?? 20
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .json)
            if !reader.isEmpty { usage(1) }

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

        case "scan-sweep":
            let nsCSV = reader.popValue(for: "--ns") ?? "1024,4096,65536,1048576"
            let iters = reader.popInt(for: "--iters") ?? 50
            let warmup = reader.popInt(for: "--warmup") ?? 10
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .jsonl)
            if !reader.isEmpty { usage(1) }

            let ns = nsCSV
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .compactMap(Int.init)

            if ns.isEmpty { die("invalid --ns '\(nsCSV)' (expected comma-separated ints)") }

            let results = try ns.map { n in
                try ScanBenchmark.run(context: context, kernels: kernels, n: n, iters: iters, warmup: warmup)
            }

            switch format {
            case .human:
                for r in results { print(r.prettyLine) }
            case .json, .jsonl:
                for r in results { print(try r.jsonLine()) }
            case .csv:
                printCSV(
                    header: ["bench", "n", "iters", "warmup", "gpu_seconds", "wall_seconds", "avg_us", "ok"],
                    rows: results.map { r in
                        [
                            "scan",
                            "\(r.n)",
                            "\(r.iters)",
                            "\(r.warmup)",
                            "\(r.gpuSeconds)",
                            "\(r.wallSeconds)",
                            "\(r.avgMicros)",
                            r.ok ? "true" : "false",
                        ]
                    }
                )
            }

        case "matmul":
            let m = reader.popInt(for: "--m") ?? 256
            let n = reader.popInt(for: "--n") ?? 256
            let k = reader.popInt(for: "--k") ?? 256
            let iters = reader.popInt(for: "--iters") ?? 50
            let warmup = reader.popInt(for: "--warmup") ?? 10
            let variantRaw = reader.popValue(for: "--variant") ?? "tiled16"
            guard let variant = MatmulVariant(rawValue: variantRaw) else {
                die("invalid --variant '\(variantRaw)' (expected naive|tiled8|tiled16|tiled32)")
            }
            let tgX = reader.popInt(for: "--tg-x")
            let tgY = reader.popInt(for: "--tg-y")
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .json)
            if !reader.isEmpty { usage(1) }

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

        case "matmul-sweep":
            let m = reader.popInt(for: "--m") ?? 512
            let n = reader.popInt(for: "--n") ?? 512
            let k = reader.popInt(for: "--k") ?? 512
            let iters = reader.popInt(for: "--iters") ?? 10
            let warmup = reader.popInt(for: "--warmup") ?? 3
            let format = OutputOptions.parse(&reader, defaultFormat: .human, jsonImplies: .jsonl)
            if !reader.isEmpty { usage(1) }

            let candidates: [(Int, Int)] = [
                (8, 8),
                (16, 8),
                (16, 16),
                (32, 8),
            ]

            var results: [MatmulResult] = []
            results.reserveCapacity(candidates.count + 3)

            for (x, y) in candidates {
                results.append(try MatmulBenchmark.run(
                    context: context,
                    kernels: kernels,
                    m: m,
                    n: n,
                    k: k,
                    iters: iters,
                    warmup: warmup,
                    variant: .naive,
                    tgX: x,
                    tgY: y
                ))
            }

            for variant in [MatmulVariant.tiled8, .tiled16, .tiled32] {
                results.append(try MatmulBenchmark.run(
                    context: context,
                    kernels: kernels,
                    m: m,
                    n: n,
                    k: k,
                    iters: iters,
                    warmup: warmup,
                    variant: variant
                ))
            }

            switch format {
            case .human:
                for r in results { print(r.prettyLine) }
            case .jsonl, .json:
                for r in results { print(try r.jsonLine()) }
            case .csv:
                printCSV(
                    header: ["bench", "variant", "tg_x", "tg_y", "m", "n", "k", "iters", "warmup", "gpu_seconds", "wall_seconds", "avg_us", "gflops", "ok"],
                    rows: results.map { r in
                        [
                            "matmul",
                            r.variant.rawValue,
                            r.tgX.map { "\($0)" } ?? "",
                            r.tgY.map { "\($0)" } ?? "",
                            "\(r.m)",
                            "\(r.n)",
                            "\(r.k)",
                            "\(r.iters)",
                            "\(r.warmup)",
                            "\(r.gpuSeconds)",
                            "\(r.wallSeconds)",
                            "\(r.avgMicros)",
                            "\(r.gflops)",
                            r.ok ? "true" : "false",
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
