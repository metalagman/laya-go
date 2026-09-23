# laya-go

`laya-go` is an embeddable Go API for evaluating one state against typed Laya
decision questions in the application process. It is designed around explicit
runtime and model ownership, ordered results, and caller-controlled model
sources.

This repository provides the public API, strict bundle contract and verifier,
locked official exporter, frozen parity corpus, and a reviewed native-runtime
candidate. An ordinary build has no native dependency and `NewRuntime` returns
`ErrNativeUnavailable`. A build made explicitly with `laya_native` and cgo can
open a complete local bundle and run in-process ONNX Runtime inference when the
pinned native libraries are supplied by the application environment.

The native path and source handling are compatibility evidence for the pinned
linux/amd64 artifacts, not a general platform-support or release claim. The
optional `adklaya` package supplies typed local-inference nodes and deterministic
choice routing for official Google ADK Go v2.4.0 workflows.

Native applications must set `ORT_DISABLE_TELEMETRY=1` before constructing a
runtime. This is the upstream pre-initialization opt-out; `laya-go` refuses to
initialize the native environment without it and does not mutate process-wide
environment state on behalf of a host application.

## Model sources and ownership

The API has two explicit source modes for a complete model bundle:

- `Runtime.OpenModelDir` accepts a caller-owned complete bundle directory. In a
  native candidate build, that directory may be an application asset or a
  complete bundle in an externally acquired Hugging Face snapshot. Ordinary
  directories reject symlinks. On the proven linux/amd64 candidate only, the
  documented direct snapshot-to-blob links are accepted. The caller keeps the
  snapshot and blob store immutable until `Model.Close` completes.
- `Runtime.OpenModelFS` accepts a caller-owned `fs.FS`, including `embed.FS`,
  and materializes the complete bundle below `FSModelOptions.Root` into the
  explicit, existing `WorkDir`. The caller keeps the filesystem readable and
  immutable until `Model.Close` completes.

Materialization streams and verifies every file, then atomically publishes one
immutable directory under `WorkDir/.laya-go/bundles/sha256-<digest>`. A valid
publication is retained and reverified for later opens; closing a Model does
not delete it. The library recovers only its exactly named, marked staging
directories and provides no pruning API. Plan free space for the complete
bundle and manage retention at the application deployment boundary while no
Models use that WorkDir. See [Source operations](docs/source-operations.md).

The library does not discover models, authenticate to model providers, select
repository revisions, download files, resolve remote filesystems, or manage an
external cache. Applications perform those operations before calling this API.

## API shape

A `State` is either exact text or validated JSON. `TextState` preserves every
byte of the Go string. `JSONState` copies, validates, and compacts caller bytes;
`State.JSON` returns another copy. Questions do not duplicate state.

```go
func predict(ctx context.Context, model *laya.Model) (laya.Prediction, error) {
	state, err := laya.JSONState([]byte(`{"customer":{"age":42}}`))
	if err != nil {
		return laya.Prediction{}, err
	}

	route, err := laya.NewChoiceQuestion(
		"route",
		"Choose a route.",
		[]laya.ChoiceCriterion{
			{ID: "accept", Description: "continue automatically"},
			{ID: "review", Description: "request human review"},
		},
	)
	if err != nil {
		return laya.Prediction{}, err
	}

	return model.Predict(ctx, state, []laya.Question{route})
}
```

`Predict` evaluates one shared state against an ordered heterogeneous question
slice. Its `Prediction` contains results in the same order and one shared
`Usage` value for the whole batch. The typed helpers `Choice`, `Score`, and
`Noul` serve single questions.

- `ChoiceResult` exposes the selected criterion, the criterion-ordered
  probability distribution, calibrated confidence, and a separate action or
  escalation probability.
- `ScoreResult` exposes the expected zero-based ordinal level, the
  rubric-ordered labeled distribution, calibrated confidence, and the separate
  action probability.
- `NoulResult` exposes calibrated `P(true)` and the separate action probability.

The action probability is not answer confidence. Truncation is opt-in through
`ModelOptions`; successful truncation is visible in `PredictionMetadata`.

## Google ADK adapter

`adklaya` accepts an already-open concrete `*laya.Model` and a typed state
projector. It does not acquire models or own runtime/model lifecycle. Data nodes
preserve question IDs, ordered distributions, `Usage`, and
`PredictionMetadata`. Choice routing emits one event containing the unchanged
typed domain input and exactly one route; unique-maximum and inclusive
acceptance thresholds select a configured branch, while ties or rejected
thresholds select the explicit fallback.

```go
node, err := adklaya.NewChoiceNode(
	model,
	func(_ agent.Context, input Request) (laya.State, error) {
		return laya.TextState(input.Text), nil
	},
	question,
	adklaya.NodeOptions{Name: "classify"},
)
```

The module pins ADK Go v2.4.0. Output DTOs support JSON and JSON Schema
generation, but nodes intentionally use the schema-less v2.4.0 FunctionNode
path because that release's explicit schema runtime rejects returned Go
structs. See [ADK operations](docs/adk-operations.md) for the characterized
boundary and runbook.

## Lifecycle, concurrency, and errors

Create a concrete `Runtime`, open concrete `Model` values, close every model,
then close the runtime:

```go
func openSnapshot(ctx context.Context, callerOwnedDirectory string) error {
	runtime, err := laya.NewRuntime(ctx, laya.RuntimeOptions{})
	if errors.Is(err, laya.ErrNativeUnavailable) {
		// Expected in an ordinary build without the native candidate.
		return err
	}
	if err != nil {
		return err
	}
	defer runtime.Close(context.Background())

	model, err := runtime.OpenModelDir(ctx, callerOwnedDirectory, laya.ModelOptions{
		QueueCapacity: 4,
	})
	if err != nil {
		return err
	}
	defer model.Close(context.Background())

	return nil
}
```

`Runtime` and `Model` are safe for concurrent use. Model admission is bounded:
one call is active in the current contract and `QueueCapacity` is the exact
number allowed to wait in FIFO order. Contexts control open, admission,
inference, and close waits and are never stored. Cancellation is reported with
the canonical `context.Canceled` or `context.DeadlineExceeded` identity.

`Model.Close` rejects new work, drains work already admitted, and releases its
resources. If its context ends first, the model remains closing and a later
call may retry. `Runtime.Close` returns `ErrModelsOpen` while a model open is in
progress or a live model remains. Successful close is concurrency-safe and
idempotent. Native inference cancellation is cooperative: the call owns its
ONNX Runtime run options, requests termination once, waits for the native call
to return before releasing tensors, and never publishes a late result.

Use `errors.Is` for stable categories such as `ErrInvalidConfig`,
`ErrInvalidBundle`, `ErrIntegrity`, `ErrUnsupportedBundle`,
`ErrUnsupportedSource`, `ErrMaterialization`, `ErrInvalidRequest`,
`ErrInputTooLong`, `ErrQueueFull`, `ErrQueueTimeout`, and
`ErrNativeUnavailable`. Use `errors.As` for `BundleError`,
`InputTooLongError`, and `QueueError` when their structured fields are needed.
Error strings are diagnostic text, not a compatibility contract.

## Compatibility and development

The module targets Go 1.26.6 or newer. Before v1, the public API may receive
additive refinements as native integration is validated; semantic changes are
reviewed explicitly rather than hidden behind mutable defaults or provider
registries.

`Taskfile.yml` is the repository runbook. Install the pinned tools, then use
`task --list` to inspect operations and `task check` to run the complete local
quality gate:

```sh
go install github.com/go-task/task/v3/cmd/task@v3.53.1
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.13.2
go install golang.org/x/vuln/cmd/govulncheck@v1.1.4
task check
```

The format and module-tidiness tasks are checks and do not rewrite files. The
vulnerability task reads the public Go vulnerability database; none of the
tasks download models, native runtimes, or application data.

The protected native runbook consumes three already-local inputs:
`LAYA_ONNXRUNTIME_LIBRARY`, `LAYA_TOKENIZERS_LIBRARY`, and `LAYA_BUNDLE_DIR`.
It supplies `ORT_DISABLE_TELEMETRY=1` to each native test process.
`task native:verify` authenticates them, `task native:scan` checks the production
dependency and source boundaries, `task native:gate` runs real-bundle parity and
lifecycle tests, `task native:gate:race` repeats the gate with the race detector,
and `task native:stress` repeats the two-model cleanup smoke. The aggregate
`task native:integration` runs all of them and needs about 4 GiB of available
memory. It does not acquire any artifact.

Source-specific operations are also discoverable: `task source:contract` runs
hermetic directory/link/materialization tests, `task source:audit` checks the
offline implementation and repository boundaries, and `task
source:integration` runs real directory, Hugging Face snapshot, `fs.FS`, race,
native regression, and memory-cleanup evidence. The protected operation uses
the same three already-local inputs and needs about 4 GiB RAM plus WorkDir
space for a complete materialized bundle; the protected HF fixture can require
one additional bundle-sized temporary copy when hard links are unavailable.

ADK-specific operations are `task adk:contract` and `task adk:audit` for
hermetic ordinary/race and boundary checks. With the same three already-local
native inputs, `task adk:gate` and `task adk:gate:race` run protected real
workflow graphs with network proxies denied; `task adk:integration` also runs
the native lifecycle cleanup stress gate.

Complete application compositions live in `examples/directory`,
`examples/embedded`, and `examples/adk`. Use `task example:contract` and `task
example:audit` for artifact-free checks; protected `task example:gate` and
`task example:gate:race` execute all three modes offline. See
[Example operations](docs/example-operations.md).

`task setup:platform` and `task setup:native` check prerequisites without
installing dependencies. `task bundle:verify` checks a caller-supplied bundle;
`task bundle:materialize-embedded` runs the opt-in real compile-time `embed.FS`
probe using an existing absolute `LAYA_EMBED_WORKDIR`; `task smoke:offline`
runs all primary native paths. `task diagnose:native` reports local identities,
while `task profile:native` writes to explicit new profile paths only. The
release distinction is important: `task release:contract` is an
artifact-independent check, whereas `task release:verify` requires protected
native inputs, an embedded work directory, a previous immutable bundle for
rollback verification, and a clean tracked checkout. `task package:linux-amd64`
archives committed source, not model weights or native libraries. See
[Release preparation](docs/release-v0.1.md) for prerequisites and recovery.
For the first v0.1.0 release, `task release:verify-initial` runs the same
protected gates but explicitly reports that rollback to a prior release cannot
be verified because no prior release exists.

Bundle format, offline export commands, and the frozen parity runbook are
documented in [Bundle format v1](docs/bundle-format-v1.md),
[Official checkpoint export v1](docs/official-export-v1.md), and
[Upstream parity v1](docs/upstream-parity-v1.md), and
[Native dependency gate](docs/native-dependencies.md), and
[Source operations](docs/source-operations.md), and
[ADK operations](docs/adk-operations.md). `task bundle:contract`
and `task reference:verify` are hermetic; protected export, regeneration, and
native integration remain explicit opt-in operations over caller-supplied local
paths.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
