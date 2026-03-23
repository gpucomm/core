# Agent Guide (gpucomm/core)

This file captures repo-specific conventions and “how we operate” notes for humans and automation.

## Build

```bash
swift build -c release
```

## CI

Workflows live under `.github/workflows/` and are documented in `docs/workflows.md`.

## Releases

### Nightly (pre-release)

The `nightly` workflow builds and publishes (updates) a **pre-release** at tag `nightly`:

- `gpucomm-macos-arm64.zip`
- `SHA256SUMS.txt`
- `results.meta`, `results.*.json`

You can also trigger it manually from GitHub Actions (workflow dispatch).

### Stable release (manual)

Stable releases are created from `main` with a version tag like `v0.1.0` and `prerelease=false`.

1) Ensure clean build:

```bash
swift build -c release
```

2) Package the binary (keep artifacts out of git; use `/tmp`):

```bash
rm -rf /tmp/gpucomm-core-release
mkdir -p /tmp/gpucomm-core-release/dist
cp ./.build/release/gpucomm /tmp/gpucomm-core-release/dist/gpucomm
(cd /tmp/gpucomm-core-release/dist && zip -9 gpucomm-macos-arm64.zip gpucomm)
shasum -a 256 /tmp/gpucomm-core-release/dist/gpucomm-macos-arm64.zip > /tmp/gpucomm-core-release/dist/SHA256SUMS.txt
```

3) Write release notes to a file (avoid shell backtick/escaping issues):

```bash
cat > /tmp/notes.md <<'EOF'
First stable release of `gpucomm/core`.
EOF
```

4) Create the GitHub Release:

```bash
TAG=v0.1.0
SHA=$(git rev-parse HEAD)
gh release create -R gpucomm/core "$TAG" \
  /tmp/gpucomm-core-release/dist/gpucomm-macos-arm64.zip \
  /tmp/gpucomm-core-release/dist/SHA256SUMS.txt \
  --target "$SHA" \
  --title "$TAG" \
  --notes-file /tmp/notes.md
```

5) Verify:

```bash
gh release view -R gpucomm/core "$TAG"
```

## Compatibility note

`Package.swift` uses `// swift-tools-version: 6.1` so GitHub Actions `macos-latest` can build reliably.

