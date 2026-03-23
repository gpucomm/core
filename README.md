# gpucomm/core

Low-level GPU compute runtime for Apple Silicon, focused on memory, synchronization, and data movement using Metal.

## Goals

- Minimal runtime: device/queue/pipeline + buffer utilities
- Kernel experiments: communication patterns + memory behavior
- Built-in benchmarks: bandwidth, latency, scaling

## Reliability On Hardware

This repo is meant to be **experiments + tests = reliability on real hardware**.

What’s solid now:

- Experiments are measurable: bandwidth/transfer/scan/matmul/latency + sweeps + `--reps` + p50/p95
- Correctness checks exist where it matters (`scan`, `matmul` for small sizes, plus `gpucomm selftest`)
- CI keeps it buildable and the CLI usable (`swift build -c release` + `gpucomm --help`)

Caveats:

- GitHub Actions macOS runners aren’t Apple Silicon GPUs you control, so CI can’t validate “real” Metal timings—only build + basic CLI behavior
- “Reliability on hardware” still depends on running these benches on target machines and tracking regressions (record chip/macOS version + commit + outputs)

## Roadmap Progress

Primary tracking issue: https://github.com/gpucomm/core/issues/1

| Milestone | Roadmap Comment | Commit |
| --- | --- | --- |
| Transfer benchmark | https://github.com/gpucomm/core/issues/1#issuecomment-4110310825 | `eefc5bb` |
| Scan (1024) | https://github.com/gpucomm/core/issues/1#issuecomment-4110321217 | `93d5627` |
| Scan (multi-block) | https://github.com/gpucomm/core/issues/1#issuecomment-4110349818 | `c7c9f9e` |
| Matmul (naive+tiled) | https://github.com/gpucomm/core/issues/1#issuecomment-4110369561 | `d42298c` |
| Matmul sweep | https://github.com/gpucomm/core/issues/1#issuecomment-4110384808 | `b70fde7` |
| Matmul tiled variants | https://github.com/gpucomm/core/issues/1#issuecomment-4110398545 | `731c33f` |
| Output formats (`--format`) | https://github.com/gpucomm/core/issues/1#issuecomment-4110425795 | `7d30ac8` |
| Scan sweep | https://github.com/gpucomm/core/issues/1#issuecomment-4110440856 | `9be8a34` |
| Bandwidth sweep | https://github.com/gpucomm/core/issues/1#issuecomment-4110451297 | `37fdb5b` |
| Transfer sweep | https://github.com/gpucomm/core/issues/1#issuecomment-4110461187 | `f000bc7` |
| Percentiles for sweeps | https://github.com/gpucomm/core/issues/1#issuecomment-4110487664 | `9054a8b` |
| `--reps` for single benches | https://github.com/gpucomm/core/issues/1#issuecomment-4110513683 | `c637e3b` |
| macOS CI build | https://github.com/gpucomm/core/issues/1#issuecomment-4110547545 | `466795e` |
| CI help smoke | https://github.com/gpucomm/core/issues/1#issuecomment-4110557042 | `25c6f84` |
| Latency benchmark | https://github.com/gpucomm/core/issues/1#issuecomment-4110584831 | `8382a5c` |
| Hardware selftest | https://github.com/gpucomm/core/issues/1#issuecomment-4110605042 | `c101034` |

## Build

```bash
swift build -c release
```

## Run

```bash
.build/release/gpucomm bench bandwidth --size-mib 64 --iters 200 --mode shared
.build/release/gpucomm bench bandwidth --size-mib 64 --iters 200 --mode private
.build/release/gpucomm bench bandwidth-sweep --sizes-mib 1,4,16,64 --iters 200 --mode private --format jsonl
.build/release/gpucomm bench scan --n 1024 --iters 200 --warmup 20
.build/release/gpucomm bench scan --n 65536 --iters 50 --warmup 10
.build/release/gpucomm bench scan-sweep --ns 1024,4096,65536,1048576 --iters 50 --warmup 10 --format jsonl
.build/release/gpucomm bench latency --kind kernel --iters 2000 --warmup 200 --reps 5 --format json
.build/release/gpucomm bench matmul --m 256 --n 256 --k 256 --iters 50 --warmup 10 --variant tiled16
.build/release/gpucomm bench matmul-sweep --m 512 --n 512 --k 512 --iters 10 --warmup 3
.build/release/gpucomm bench transfer --size-kib 4 --iters 10000 --warmup 100 --direction h2d --mode private --strategy blit
.build/release/gpucomm bench transfer --size-kib 4 --iters 10000 --warmup 100 --direction d2h --mode private --strategy blit --format json
.build/release/gpucomm bench transfer-sweep --sizes-kib 1,4,64 --iters 5000 --warmup 200 --direction both --mode both --format jsonl
.build/release/gpucomm run reduction --n 1024
.build/release/gpucomm selftest
```

## Layout

- `Sources/GPUCommCore`: runtime + benchmarks
- `Sources/GPUCommCore/Resources/Kernels`: Metal kernels (compiled at runtime)
- `Sources/gpucomm`: CLI
