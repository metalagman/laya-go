# Story 4 evidence: bundle sources and safe materialization

This document maps Story 4 requirements to durable repository evidence. It
covers already-local complete Bundle v1 sources and the pinned linux/amd64
native candidate. It does not implement acquisition, authenticate to Hugging
Face, manage an external cache, claim another platform, or modify
`pack/callee/**`.

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| REQ-SOURCE-001, REQ-DIR-001/002 | `internal/source/source.go`, `native_lifecycle.go` | one verified source handle and native-open path; exact ordinary-directory verification before native allocation; immutable caller ownership |
| REQ-LINK-001 | `internal/source/snapshot.go` | exact linux/amd64 snapshot/blob classifier; direct relative regular-blob links; malformed, escaping, cyclic, dangling, multi-hop, and mutable cases fail closed |
| REQ-FS-001, REQ-MAT-001/002 | `internal/source/stage.go`, `internal/source/materialize.go` | `fs.ValidPath` root, strict preverification, 64 KiB bounded streaming, exact size/digest checks, deterministic fully reverified retained publication |
| REQ-ATOMIC-001, REQ-CONCURRENT-001 | `internal/source/materialize.go`, `lock_linux.go` | same-filesystem marked staging, sync, no-replace atomic rename, advisory process lock, goroutine/process contention and every-checkpoint fault tests |
| REQ-RECOVERY-001 | `recoverStages` and materialization tests | recovery deletes only exact-name, exact-marker stages; malformed, symlinked, unrelated, caller, and provider paths are preserved |
| REQ-LEASE-001 | `native_lifecycle.go`, source/native lifecycle tests | Model retains the source; reverse session/tokenizer/source teardown; independent shared publication leases; retryable close |
| REQ-ERROR-001 | `errors.go`, `internal/source.Error`, error tables | stable invalid/integrity/unsupported/materialization/context/native identities; safe causes and redacted paths/content |
| REQ-OFFLINE-001, REQ-MAINTAIN-001 | concrete public methods, `native:scan`, `source:audit` | no provider abstraction, HTTP/gRPC client, Python, child process, credential, resolver, downloader, remote filesystem, or cache manager |
| REQ-QUALITY-001, REQ-DOC-001 | Taskfile, package docs, README, bundle/source runbooks | hermetic contract/race checks; explicit protected inputs; raw/complete, source, WorkDir, disk, cancellation, cleanup, and rollback boundaries |

## Protected observations

On 2026-09-22, the source gate used the exact protected Story 2/3 inputs and
accepted BundleID
`sha256:b3d35e00b0689988dff23f0eeed8721f10cce2c3a10d720f6c7d42cfc7d8c4de`
in every source mode.

The final non-race `task source:gate` completed in 97.352 seconds after the test
fixture was changed to require no write beside the caller-owned source. The
ordinary directory, exact HF snapshot/blob topology, first `fs.FS`
materialization, and retained reuse each passed all 14 frozen public cases and
three raw batches. Cancellation recovery and two concurrent Models sharing one
publication also passed.

`task source:gate:race` passed the same source/lifecycle cases under the race
detector. `task native:integration` then passed directory regression, native
race, and two consecutive two-model RSS cleanup cycles. The source gates ran
with `GOPROXY=off`, Hugging Face/Transformers offline flags, denied HTTP(S)/all
proxies, and ONNX telemetry disabled. No network or child-process production
path and no `:memory:.ses` artifact was observed.

Hermetic source unit and race suites cover `embed.FS`, `fstest.MapFS`, hostile
readers, bounded large streaming, process contention, lock cancellation,
publication tampering, every injected materialization boundary, and narrowly
owned recovery. Ordinary format, tidy, module, unit, race, and lint gates pass
without native artifacts. The dependency set is the pinned ONNX binding plus
BSD-3-Clause `golang.org/x/sys` for the Linux lock/atomic-rename primitives.
No model/native artifact over 10 MiB is stored in the project tree.

These observations are candidate evidence only. Publication or broader support
requires the later Epic validation and explicit release decision.
