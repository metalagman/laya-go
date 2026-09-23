# ADK operations

The optional `adklaya` package adapts one already-open concrete `*laya.Model`
to official Google ADK Go workflows. It supports typed Choice, Score, Noul, and
ordered mixed data nodes plus deterministic choice routing. The application
still owns model acquisition, the immutable complete bundle, `Runtime`,
`Model`, and their close order. The adapter never opens, configures, or closes
either object.

`StateProjector[IN]` is the only boundary between an application's typed
workflow value and `laya.State`. Every node calls `Model.Predict`, including
single-question nodes, so `Usage` and `PredictionMetadata` remain available.
Errors and cancellation stop the graph; they are never converted into a
fallback route.

Choice routing uses an explicit non-nil `AcceptancePolicy`. A branch is selected
only when the selected criterion is the unique maximum and all three inclusive
probability, confidence, and action thresholds pass. Exact maximum ties and
threshold rejection take the one explicit fallback edge. Multiple criteria
that share a successor are represented by one ADK `MultiRoute`, so a successor
cannot run twice. The routing event forwards the original typed domain input
unchanged; Laya's choice is used only to select the prevalidated route.

## ADK v2.4.0 boundary

The module pins `google.golang.org/adk/v2 v2.4.0`. The exported output structs
are JSON-tagged and support JSON Schema generation. They are intentionally
installed in schema-less `workflow.FunctionNode` values for this exact ADK
version: its explicit schema path passes a returned Go struct directly to
`jsonschema-go`, which rejects structs at runtime. The executable
`internal/adkcontract` suite freezes that behavior. Revisit this choice only
after pinning and characterizing a newer official ADK release.

The official ADK module has a broad transitive dependency closure that includes
optional network-capable packages. `adklaya` itself imports no HTTP, gRPC,
Python, or child-process client and exposes no remote inference or acquisition
path. Protected tests deny proxies and consume only already-local artifacts.

## Runbook

`task adk:contract` runs the pinned framework characterization and all hermetic
adapter graphs in ordinary and race modes with `GOPROXY=off`. `task adk:audit`
adds source, lifecycle, secret, license, and protected-path checks.

Protected operations require the same already-local
`LAYA_ONNXRUNTIME_LIBRARY`, `LAYA_TOKENIZERS_LIBRARY`, and `LAYA_BUNDLE_DIR` as
the native runbook. `task adk:gate` runs real typed and routing graphs with
network proxies denied; `task adk:gate:race` repeats them under the race
detector. `task adk:integration` also runs the native lifecycle/RSS stress gate.
None of these operations downloads a model or native library.

The protected gate loads the roughly 1.3 GiB FP32 bundle and should be given
about 4 GiB of available memory. Race and lifecycle stress operations take
longer and may retain allocator pages temporarily even after native resources
are released.

## Troubleshooting and rollback

- A missing-input failure means one of the three protected environment paths
  is unset or no longer names the reviewed local artifact. Run `task
  native:verify` before the ADK gate.
- `ErrNativeUnavailable` means the binary was not built with the protected
  native tag/cgo setup or the ONNX Runtime path is absent. Hermetic ADK tests do
  not need native artifacts.
- A route going to fallback is a successful but unaccepted prediction, not an
  inference failure. Inspect the typed output's selected probability,
  confidence, and action probability. Errors and cancellation produce no route.
- A schema-validation failure after changing node construction is likely the
  pinned v2.4.0 raw-struct limitation. Run `task adk:contract` and do not enable
  explicit runtime schemas without first characterizing a newer pinned release.

The adapter is optional. Rollback consists of removing graph use and calling
the same caller-owned `Model` directly; bundle format, source handling, native
inference, and model lifecycle remain unchanged. Do not replace a rejected
route with unconditional/default edges and do not weaken offline gates.

The protected graph asserts the accepted immutable bundle identity
`sha256:963bc035d885e0463ea1d7d54906cadf0e2a3a41073e131921800ea5cc358935`.
