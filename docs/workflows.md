# Workflows

This repo uses GitHub Actions for build health and for publishing a nightly pre-release.

## `ci.yml`

File: `.github/workflows/ci.yml`

Purpose:

- Ensures the repo builds on GitHub’s macOS runners
- Runs a CLI smoke check that does **not** require a Metal device

What it runs:

- `swift build -c release`
- `./.build/release/gpucomm --help`

Notes:

- CI is not a performance signal. It’s only a build + CLI sanity check.

## `nightly.yml`

File: `.github/workflows/nightly.yml`

Triggers:

- Scheduled (daily) and manual (`workflow_dispatch`)

Purpose:

- Build a release binary
- Run lightweight hardware sanity checks (best-effort)
- Publish/update a GitHub **pre-release** at tag `nightly`

Artifacts/Assets:

- `gpucomm-macos-arm64.zip` (contains `gpucomm`)
- `SHA256SUMS.txt`
- `results.meta` + `results.*.json` (selftest + quick bench outputs)

Notes:

- GitHub’s hosted runners are not your target machines. Treat the nightly outputs as “smoke + trends”, not authoritative performance numbers.
- The workflow uses `contents: write` to update the `nightly` pre-release.

