# Story 5 evidence: ADK adapters

- Official dependency: `google.golang.org/adk/v2 v2.4.0`, with its module
  Apache-2.0 license present in the Go module cache.
- Hermetic characterization: direct `Event{Output, Routes}`, runner sequencing,
  duplicate-edge rejection, `MultiRoute`, retry, timeout, cancellation,
  concurrency, and the v2.4.0 struct-schema limitation.
- Adapter coverage: Choice, Score, Noul, ordered mixed output, JSON/schema
  generation, copied configuration and edges, exact routing thresholds, ties,
  fallback, errors, whole-attempt retry, timeout, and concurrent sessions.
- Protected native input: bundle
  `sha256:b3d35e00b0689988dff23f0eeed8721f10cce2c3a10d720f6c7d42cfc7d8c4de`,
  ONNX Runtime 1.29.0, and the reviewed local tokenizer archive.
- Protected result: real mixed and choice-routing ADK graphs passed in ordinary
  and race modes with offline environment flags and denied proxies. The caller
  opened and closed the shared model/runtime; adapter nodes did not own their
  lifecycle.

Executable operations are `task adk:contract`, `task adk:audit`, `task
adk:gate`, `task adk:gate:race`, and `task adk:integration`.
