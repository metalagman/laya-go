# Laya Go Bundle v1

Bundle v1 is the sole model input accepted by the laya-go runtime. A raw
Safetensors checkpoint is not a bundle and cannot be passed to
`Runtime.OpenModelDir` or `Runtime.OpenModelFS`.

## Layout

A complete bundle contains `manifest.json` at its root and exactly the regular
files declared by the manifest. Required logical roles are:

- one complete `model.onnx` graph;
- zero or more explicitly enumerated ONNX external-data files;
- `tokenizer/tokenizer.json` and `tokenizer/tokenizer_config.json`;
- `rl_agent_config.json`;
- the model and SDK-derived-code license/notice files required by the export
  profile.

The manifest itself is not included in its file table. Every declared artifact
has an exact byte size and lowercase SHA-256. Undeclared, missing, duplicate,
case-colliding, escaping, or modified files invalidate the bundle before native
loading.

## Identity and compatibility

BundleID is `sha256:` followed by the SHA-256 of the exact `manifest.json`
bytes. The exporter emits deterministic UTF-8 JSON, stable ordering, and an
explicit caller-supplied creation epoch. No wall-clock or machine-local path is
part of the identity.

Manifest v1 supports only preprocessing v1, the accepted
`laya-multilingual-v1` profile, opset 18, FP32, and CPUExecutionProvider. An
unknown schema, profile, preprocessing version, precision, opset, tensor
contract, or runtime bound fails as an unsupported bundle; it never selects a
fallback.

The complete forward contract is ordered:

1. `input_ids`: int64 `[batch, sequence]`
2. `attention_mask`: int64 `[batch, sequence]`
3. `marker_pos`: int64 `[batch, markers]`
4. `marker_mask`: bool `[batch, markers]`
5. `qtype`: int64 `[batch]`

Outputs are `logits` float32 `[batch, markers]` and `act_logits` float32
`[batch, actions]`.

## Source neutrality

The same logical bundle bytes may live in a caller-owned directory, a directory
populated by external Hugging Face tooling, or an `fs.FS` such as `embed.FS`.
The manifest contains no repository URL, credential, cache root, WorkDir, or
materialization lease. Acquisition and source-specific path safety are outside
this format.

Directory callers must keep a prepared bundle immutable for the Model lifetime.
Ordinary directories reject links. On the proven linux/amd64 candidate, an
eligible Hugging Face cache snapshot may use direct relative links to regular,
content-addressed files in its sibling `blobs` directory; all other link forms
fail closed.

An `fs.FS` source is verified below its explicit root and streamed into an
immutable, digest-addressed publication below the caller's WorkDir. That
materialization is retained and reverified for reuse; it is not acquisition or
external cache management. The complete ownership, disk, cancellation,
recovery, and rollback contract is in [Source operations](source-operations.md).
