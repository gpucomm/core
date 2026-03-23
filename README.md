# gpucomm/core

Low-level GPU compute runtime for Apple Silicon, focused on memory, synchronization, and data movement using Metal.

## Goals

- Minimal runtime: device/queue/pipeline + buffer utilities
- Kernel experiments: communication patterns + memory behavior
- Built-in benchmarks: bandwidth, latency, scaling

## Build

```bash
swift build -c release
```

## Run

```bash
.build/release/gpucomm bench bandwidth --size-mib 64 --iters 200 --mode shared
.build/release/gpucomm bench bandwidth --size-mib 64 --iters 200 --mode private
.build/release/gpucomm bench transfer --size-kib 4 --iters 10000 --warmup 100 --direction h2d --mode private --strategy blit
.build/release/gpucomm bench transfer --size-kib 4 --iters 10000 --warmup 100 --direction d2h --mode private --strategy blit --json
.build/release/gpucomm run reduction --n 1024
```

## Layout

- `Sources/GPUCommCore`: runtime + benchmarks
- `Sources/GPUCommCore/Resources/Kernels`: Metal kernels (compiled at runtime)
- `Sources/gpucomm`: CLI
