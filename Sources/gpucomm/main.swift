import Foundation
import GPUCommCore

private func usage(_ exitCode: Int32) -> Never {
    let text = """
    gpucomm — Metal-based GPU communication + compute experiments

    Usage:
      gpucomm bench bandwidth [--size-mib N] [--iters N] [--mode shared|private]
      gpucomm bench transfer [--size-kib N] [--iters N] [--warmup N] [--direction h2d|d2h] [--mode shared|private] [--strategy memcpy|blit] [--json]
      gpucomm run reduction [--n N]

    Examples:
      gpucomm bench bandwidth --size-mib 64 --iters 200 --mode shared
      gpucomm bench bandwidth --size-mib 64 --iters 200 --mode private
      gpucomm bench transfer --size-kib 4 --iters 10000 --warmup 100 --direction h2d --mode private --strategy blit
      gpucomm bench transfer --size-kib 4 --iters 10000 --warmup 100 --direction d2h --mode private --strategy blit --json
      gpucomm run reduction --n 1024
    """
    print(text)
    Foundation.exit(exitCode)
}

private func die(_ message: String, exitCode: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data(("error: \(message)\n").utf8))
    Foundation.exit(exitCode)
}

private struct ArgReader {
    private var args: [String]
    private var index: Int = 0

    init(_ args: [String]) { self.args = args }

    var isEmpty: Bool { index >= args.count }

    mutating func pop() -> String? {
        guard index < args.count else { return nil }
        defer { index += 1 }
        return args[index]
    }

    mutating func popValue(for flag: String) -> String? {
        guard index < args.count, args[index] == flag else { return nil }
        index += 1
        return pop()
    }

    mutating func popInt(for flag: String) -> Int? {
        guard let value = popValue(for: flag) else { return nil }
        return Int(value)
    }

    mutating func popFlag(_ flag: String) -> Bool {
        guard index < args.count, args[index] == flag else { return false }
        index += 1
        return true
    }
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
            if !reader.isEmpty { usage(1) }

            let result = try BandwidthBenchmark.run(
                context: context,
                kernels: kernels,
                sizeBytes: sizeMiB * 1024 * 1024,
                iters: iters,
                mode: mode
            )
            print(result.prettyLine)

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
            let json = reader.popFlag("--json")
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
            if json {
                print(try result.jsonLine())
            } else {
                print(result.prettyLine)
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
