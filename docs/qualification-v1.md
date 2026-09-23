# Qualification v1

## Pinned linux/amd64 evidence boundary

The v0.1.0 evidence covers only the reviewed linux/amd64 FP32 native
configuration with ONNX Runtime 1.29.0, the pinned tokenizer archive, and bundle
`sha256:963bc035d885e0463ea1d7d54906cadf0e2a3a41073e131921800ea5cc358935`.
Frozen multilingual/boundary preprocessing, tensor, numeric, and typed-result
parity pass the committed tolerances. Directory, HF snapshot-link, fs.FS,
native lifecycle, cancellation, race, RSS cleanup, ADK, and application example
gates pass offline.

The [clean-checkout native qualification run](https://github.com/metalagman/laya-go/actions/runs/35848824434)
passed on a GitHub-hosted Ubuntu 24.04 linux/amd64 runner. This is evidence
for the exact pinned configuration, not a blanket claim for Linux distributions
or untested environments. The `Native qualification` workflow is manual,
runs only for `main`, and uses a GitHub-hosted runner. It fetches the exact
pinned official checkpoint files from Hugging Face and the pinned reference
SDK archive from GitHub, verifies
their digests, then runs `task bundle:export` with the locked build-time tools
and network disabled. It verifies the resulting logical bundle ID and obtains
the two native libraries from their pinned official release archives. Archive
and native-file digests and ABI are verified before the protected gates run
offline. Missing, wrong, or unavailable inputs fail the job; they do not turn
skipped tests into support evidence. The bundle remains a disposable CI
fixture: no model weights are committed or attached to the library release.
No self-hosted runner, fork pull request, or runtime model downloader is
involved.

No other OS/architecture, quantized/optimized precision, execution provider,
or parallel native worker is supported. Archive availability or successful
cross-compilation is not support evidence.

## Performance method

`task qualification:gate` runs 20 sequential two-choice predictions after one
cold Runtime and Model open. It reports the host, source-model and bundle
identity, precision, pinned ONNX Runtime version and library digest, logical
CPU count, GOMAXPROCS, process threads and RSS after load, cold-open durations,
p50/p95 latency, and sequential throughput. ONNX Runtime uses its default
thread settings; the library does not expose a thread-count knob. A separate
controlled probe holds the serialized inference slot for 25 ms and measures
one queued caller's admission delay. This is a queue-behavior check, not a
service-load estimate. The task disables telemetry, denies proxies, and does
no acquisition. Results describe the named host/run only and are not a service
SLO.

The first three observations below used the earlier local manifest
`sha256:b3d35e00b0689988dff23f0eeed8721f10cce2c3a10d720f6c7d42cfc7d8c4de`.
The graph and external-data SHA-256 values match the canonical bundle, but its
exporter Git revision named a bootstrap commit, so it is not the release
artifact identity.

Observed 2026-09-23 in the current linux/amd64 qualification container (4
logical CPUs): Runtime open 36.242 ms, Model open 3156.709 ms, loaded RSS 639
MiB, 10 process threads, 20 sequential predictions, p50 45.798 ms, p95 55.365
ms, and 21.077 predictions/s. Re-run the gate on deployment hardware rather
than extrapolating these container-local measurements.

A second local run on 2026-09-23 (EliteBook, 4 logical CPUs, GOMAXPROCS 4,
ONNX Runtime 1.29.0 default threads, same FP32 bundle) measured Runtime open
112.528 ms, Model open 8248.075 ms, loaded RSS 866 MiB, 10 process threads,
controlled queue admission delay 25.194 ms with a 25 ms hold, p50 256.629 ms,
p95 305.309 ms, and 3.869 sequential predictions/s. The difference from the
container result reinforces that these are host/run observations, not portable
capacity or latency guarantees.

An immediate repeat on the same host reported Runtime open 33.701 ms, Model
open 3011.709 ms, loaded RSS 641 MiB, queue probe 25.390 ms, p50 40.718 ms,
p95 42.836 ms, and 25.328 sequential predictions/s. The run-to-run spread
must be retained rather than averaged into a deployment promise.

The canonical bundle `sha256:963bc035d885e0463ea1d7d54906cadf0e2a3a41073e131921800ea5cc358935`
on the same EliteBook measured Runtime open 38.581 ms, Model open 3059.707 ms,
loaded RSS 634 MiB, queue admission delay 29.419 ms with a 25 ms hold, p50
111.912 ms, p95 181.740 ms, and 8.914 sequential predictions/s. This is
another local observation, not a deployment capacity claim.

The clean-checkout GitHub-hosted Ubuntu 24.04 runner (4 logical CPUs) measured
Runtime open 36.356 ms, Model open 2299.804 ms, loaded RSS 634 MiB, queue
admission delay 25.586 ms with a 25 ms hold, p50 40.321 ms, p95 42.373 ms,
and 24.554 sequential predictions/s. This is runner-local evidence only;
it does not establish deployment latency or throughput.

## Application quality status

Parity is implementation equivalence, not application suitability. This
repository contains no provenance-bound labeled calibration and separate
holdout dataset for Choice, Score, Noul, or routing business outcomes.
Consequently no accuracy, calibration, fairness, route-quality, or production
suitability claim is made. Such a claim requires an independently versioned
dataset/report with leakage controls; LLM agreement, file size, latency, and
upstream parity are explicitly insufficient substitutes.

## Reproduction and rollback

Run `task check`, `task source:integration`, `task adk:integration`, `task
example:integration`, and `task qualification:gate` with the three explicit
already-local protected paths. A regression withdraws only the affected claim;
it must not broaden tolerances or relabel an untested target as supported.
