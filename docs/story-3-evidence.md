# Story 3 evidence: local native runtime candidate

This document maps Story 3 to durable repository evidence. It covers an
explicit `laya_native` cgo build using already-local pinned artifacts. It does
not declare general platform support, package native binaries, acquire models,
implement `fs.FS` materialization, or change the public API contract.

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| Pinned native stack | `internal/native/deps/manifest.json`, `scripts/verify-native-artifacts.sh`, `internal/native/tokenizer` | exact release, archive, library, license, source, ABI, and version checks; local-only tokenizer surface |
| Runtime ownership | `native_backend_enabled.go`, `native_lifecycle.go` | process-shared compatible environment, required pre-init telemetry opt-out, reference-counted leases, retryable reverse-order teardown |
| Model loading | `native_lifecycle.go`, `internal/bundle` | verified complete caller-owned directory, external-data graph, tokenizer and session construction, unwind tests for every failure boundary |
| Inference contract | `preprocess.go`, `calibration.go`, `postprocess.go`, `native_backend_enabled.go` | exact five int64 inputs and two float32 outputs, shape/dtype/count checks, stable typed results and metadata |
| Upstream parity | `native_integration_test.go`, `internal/parity/testdata/corpus-v1.json` | all 14 public cases plus all three raw batches within the committed absolute/relative tolerances |
| Queue and lifecycle | `model.go`, `model_test.go`, `native_lifecycle_test.go` | bounded FIFO admission, queue timeout/cancellation, close drain/retry, concurrent environment leases |
| Cancellation safety | `native_run.go`, `native_run_test.go` | call-owned run options, one termination request, wait-before-free, late-result discard, subsequent-run recovery |
| Offline boundary | `Taskfile.yml`, dependency/source scans | no production network-client, model-acquisition, Python, child-process, cache, authentication, or provider-registry capability |
| Quality and maintenance | `Taskfile.yml`, `repository_test.go`, `docs/native-dependencies.md` | ordinary checks remain native-independent; protected artifact, race, parity, stress, and no-telemetry gates are discoverable |

## Protected observations

On 2026-09-22, the protected runbook used the Story 2 bundle and the exact
native artifacts recorded in the dependency manifest:

- bundle: `sha256:b3d35e00b0689988dff23f0eeed8721f10cce2c3a10d720f6c7d42cfc7d8c4de`;
- ONNX Runtime 1.29.0 library:
  `5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca`;
- tokenizers release v1.27.0 static library, reporting FFI 1.26.0:
  `e6862b31745bb7d07980fcee70e49cd3b4318097609180f5d2d3fb394f305d50`.

`task native:integration` passed artifact verification, offline scans, real
parity, cancellation recovery, queue/lifecycle behavior, the race detector,
and two consecutive two-model stress cycles. In the final diagnostic run, RSS
moved from a 9 MiB baseline to 55 MiB after the first complete teardown, then
from 55 MiB to 59 MiB after the second. Both were within the 128 MiB-over-
baseline bound. The trim makes allocator caching observable; it is not part of
production behavior and does not broaden the candidate platform claim.

`task check` passed independently of all native artifacts. No native library or
model weight is stored in the repository, and no public source-acquisition or
provider-specific API was added. A caller may supply a bundle prepared from an
embedded application asset or from an externally downloaded/cached Hugging Face
snapshot; acquisition remains caller-owned.
