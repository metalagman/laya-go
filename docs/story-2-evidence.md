# Story 2 evidence: bundle contracts and upstream parity

This document maps the completed Story 2 requirements to durable repository
evidence. It does not claim that the later native runtime, source
materialization, ADK adapter, release packaging, or any platform is supported.

| Requirement | Code and contract | Hermetic evidence |
| --- | --- | --- |
| Provenance and profile | `tools/export/profiles/laya-multilingual-v1.json`, `tools/export/src/laya_export/provenance.py` | profile/source/SDK mismatch and exact-digest tests |
| Manifest v1 | `schema/bundle-manifest-v1.schema.json`, `internal/bundle` | strict parse, compatibility, fuzz, path, digest, and source-neutral layout tests |
| Reference semantics | `tools/export/src/laya_export/reference.py`, `docs/upstream-parity-v1.md` | locked transformation/property tests and pinned SDK semantic probe |
| Official export | `architecture.py`, `onnx_export.py`, `exporter.py` | exact 170-tensor load, five-input/two-output shape and dtype checks, reference/ONNX parity, atomic failure tests |
| Frozen fixtures | `internal/parity/testdata/corpus-v1.json`, `internal/parity` | exact corpus digest, strict loader, inventory, tensor, tolerance, ordering, usage, and tamper tests |
| Source neutrality | `internal/bundle/testdata`, `docs/bundle-format-v1.md` | identical ordinary, Hugging Face snapshot-shaped, MapFS, and embedded bundle verification |
| Errors and operations | `errors.go`, `internal/bundle/errors.go`, `Taskfile.yml` | `errors.Is`/`errors.As` tables and discoverable hermetic versus protected tasks |
| Scope and security | package dependency/process scans and contract tests | no production Python/process/network/acquisition/provider code, no community oracle, no large model artifact |

## Protected artifact evidence

On 2026-09-22, the pinned official checkpoint and SDK were supplied as local
inputs and verified before construction. Two locked exports were byte-identical:

- BundleID: `sha256:b3d35e00b0689988dff23f0eeed8721f10cce2c3a10d720f6c7d42cfc7d8c4de`
- graph: opset 18, 1,486 nodes, exact dynamic batch/sequence/marker shapes,
  five named inputs and two named outputs;
- checkpoint coverage: 170 tensors and 321,908,995 parameters, with all keys
  and shapes consumed exactly;
- smoke parity: maximum absolute difference `9.179115295410156e-06` for
  `logits` and `0.0008544921875` for `act_logits`, passing the frozen
  `atol=1e-4`, `rtol=1e-4` comparison.

The 14-positive/2-negative corpus was then generated twice from that bundle
with identical bytes:

- corpus SHA-256: `4e08c3230a4ed73ae54e09a81411c2c427e8766e3395867dcc5d8d0db26fc677`;
- three ordered runs: mixed-language batch, short coverage batch, and exact
  fit/right-truncation boundary batch;
- no private input, credential, community identity, or model artifact.

The protected artifacts remain outside Git. The repository retains only the
76,981-byte synthetic corpus and tiny synthetic bundle fixtures. Reproduction
uses the explicit offline commands in `docs/official-export-v1.md`; absence of
the official local inputs is an actionable failure, never a skipped success.
