# Source operations

This runbook covers local sources accepted by the current `laya_native` cgo
candidate. It is evidence for pinned linux/amd64 artifacts, not a general
platform-support claim. Every operation starts with bytes already supplied by
the caller; `laya-go` has no provider client, authentication, revision
resolver, downloader, remote filesystem, or external cache manager.

## Choose the runtime input

A raw Safetensors checkpoint is input to the separate build-time exporter. It
is not a runtime bundle. Both public open methods require one complete Bundle
v1 with `manifest.json`, the ONNX graph and any external data, tokenizer files,
calibration, and required license/notice files. The strict allow-list, sizes,
digests, compatibility, and BundleID are defined in
[Bundle format v1](bundle-format-v1.md).

Use `Runtime.OpenModelDir` when a complete bundle already has stable concrete
paths. Use `Runtime.OpenModelFS` for an `fs.FS`, including `embed.FS`, when the
native graph first needs concrete local paths. Both paths verify the same
Bundle v1 and enter the same tokenizer/session constructor.

## Caller-owned directories

An ordinary bundle directory may contain only directories and regular files;
symlinks and special files fail before native allocation. The library opens a
root-confined handle, verifies identity and every manifest entry, and never
writes to the source. The caller must keep the directory immutable until every
Model using it has closed.

On the current linux/amd64 candidate, `OpenModelDir` also recognizes this exact
Hugging Face snapshot shape:

```text
models--<owner>--<repository>/
├── blobs/
│   └── <40-or-64-lowercase-hex>
└── snapshots/
    └── <40-or-64-lowercase-hex-revision>/
        ├── manifest.json
        ├── model.onnx -> ../../blobs/<lowercase-hex>
        └── ...
```

Any linked bundle file must be one direct relative symlink to one regular blob
in that repository's sibling `blobs` directory. The snapshot and blob roots
must be real directories. Absolute, unclean, dangling, directory, cyclic,
multi-hop, non-hex, or outside targets fail closed. Manifest size and SHA-256
verification still applies. The library neither mutates nor locks a provider
cache; the caller must keep the snapshot, links, and blobs immutable through
`Model.Close`.

The special link policy is intentionally unavailable on unproven platforms.
Copy or otherwise prepare a complete symlink-free directory there; this does
not by itself make the native backend supported on that platform.

## `fs.FS` materialization

`FSModelOptions.Root` selects the complete bundle using slash-separated
`fs.ValidPath` rules; empty means `.`. `FSModelOptions.WorkDir` must name an
existing, caller-owned, writable real directory. It must not be a symlink.
Keep the source filesystem readable and immutable until `Model.Close` returns.

The library creates only this private namespace:

```text
WorkDir/.laya-go/
├── owner-v1
├── materialize.lock
├── staging/
│   └── <bundle-digest>-<random>/
└── bundles/
    └── sha256-<bundle-digest>/
```

Before copying, the source is verified. Each declared file is then streamed
through a fixed 64 KiB buffer into an exclusively created stage. Writes are
bounded by the manifest size and checked for exact size and SHA-256. Files and
directories are synchronized, the staged bundle is verified again, and one
no-replace rename publishes it on the same WorkDir filesystem. A reader sees a
complete publication or no publication, never a partial ready tree.

The published directory is deterministic for the exact BundleID. Later opens
serialize through the namespace lock and fully reverify it before reuse. A
tampered ready publication is preserved and rejected; the library does not
silently replace evidence of corruption.

Plan WorkDir free space for at least one complete bundle plus small staging
metadata. Distinct BundleIDs accumulate as distinct retained publications.
There is deliberately no pruning or public materialization-cache API. An
operator may retire the entire application-owned WorkDir only after stopping
all runtimes and Models that can use it; `laya-go` never performs that broad
deletion.

## Cancellation, recovery, and ownership

Context cancellation is checked during verification, lock wait, copying,
synchronization, and open. A failed call removes only the uniquely named stage
it created. On the next open, recovery removes only an exactly named staging
directory containing the exact library ownership marker. Malformed, symlinked,
or unrelated WorkDir entries are not recursively consumed as library state.

A successful Model owns its source lease. Native close releases the ONNX
session, then tokenizer, then source handle. Closing one Model does not remove
or invalidate a retained publication used by another. Close each Model before
its Runtime; a canceled close may be retried.

Error handling uses stable identities:

- `ErrInvalidBundle` covers malformed or incomplete input;
- `ErrIntegrity` covers size, digest, or observable identity mismatch;
- `ErrUnsupportedBundle` covers valid but incompatible bundle metadata;
- `ErrUnsupportedSource` covers unsupported link/layout/platform semantics;
- `ErrMaterialization` covers WorkDir, staging, locking, sync, and publication
  I/O;
- `context.Canceled` and `context.DeadlineExceeded` retain their standard
  identity.

Use `errors.Is`, and `errors.As` to `BundleError` where structured bundle fields
are useful. Diagnostic strings omit external absolute paths and file content.

## Operations

No source operation acquires a model or native dependency.

```sh
# Hermetic bundle-source, link, staging, publication, process-contention,
# recovery, cancellation, and race contracts. No native artifacts required.
task source:contract

# Offline API/dependency/license/secret/large-file/protected-path audit.
task source:audit

# Verify an already-local complete bundle.
LAYA_BUNDLE_DIR=/absolute/path/complete-bundle task bundle:verify
```

The protected source gate additionally needs the already-local pinned native
libraries and complete official bundle:

```sh
export LAYA_ONNXRUNTIME_LIBRARY=/absolute/path/libonnxruntime.so.1.29.0
export LAYA_TOKENIZERS_LIBRARY=/absolute/path/libtokenizers.a
export LAYA_BUNDLE_DIR=/absolute/path/complete-bundle
task source:integration
```

`source:gate` runs real inference from an ordinary directory, a generated exact
snapshot/blob topology, first `fs.FS` materialization, retained reuse,
cancellation recovery, and shared-publication close ordering. It denies network
proxies and uses `GOPROXY=off`. `source:gate:race` repeats it under the race
detector. `source:integration` also runs the complete native regression and RSS
cleanup suite; allow about 4 GiB RAM and, when hard links are unavailable, up
to two complete bundle sizes across temporary and WorkDir storage. Missing
inputs fail with actionable preconditions rather than skipping.

## Failure response and rollback

- Invalid or incompatible source: preserve it, select a previously verified
  immutable complete bundle, and rerun `bundle:verify` and the protected gate.
- Interrupted materialization: retry the same open. Recovery can remove only a
  marked abandoned stage; it never deletes the source, provider cache, WorkDir,
  or an existing publication.
- Corrupt retained publication: stop every Model using that WorkDir, preserve
  the rejected directory for investigation, select a clean WorkDir or remove
  that exact application-owned namespace under operator control, then
  rematerialize and reverify. The library will not overwrite it.
- Native regression: roll back the application, pinned native libraries, and
  BundleID as one immutable set. Never replace an existing release tag or
  mutate an external snapshot in place.
