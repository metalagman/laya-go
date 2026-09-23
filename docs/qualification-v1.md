# Qualification v1

## Local candidate evidence boundary

The current local evidence covers only the reviewed linux/amd64 FP32 native
candidate with ONNX Runtime 1.29.0, the local tokenizer archive, and bundle
`sha256:b3d35e00b0689988dff23f0eeed8721f10cce2c3a10d720f6c7d42cfc7d8c4de`.
Frozen multilingual/boundary preprocessing, tensor, numeric, and typed-result
parity pass the committed tolerances. Directory, HF snapshot-link, fs.FS,
native lifecycle, cancellation, race, RSS cleanup, ADK, and application example
gates pass offline.

This is not yet an advertised supported-platform claim: the native GitHub
Actions job has not completed against a clean committed checkout. The
`Native qualification` workflow is manual, runs only for `main`, and uses a
GitHub-hosted runner. A trusted operator must set the repository secret
`LAYA_CI_BUNDLE_URL` to an HTTPS URL for a root-layout bundle archive, then
provide that archive's SHA-256 as the workflow input. The URL may use
short-lived authentication; it is never supplied as a public workflow input or
printed by the job.
Do not attach the model bundle to the library's source release. The workflow
obtains the two native libraries from their pinned official release archives.
It verifies archive digests before extraction, then native file digests/ABI
and the expected logical bundle ID,
and runs the protected gates offline. Missing, wrong, or unavailable artifacts
fail the job; they do not turn skipped tests into support evidence. No self-hosted
runner, fork pull request, or runtime model downloader is involved.

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
