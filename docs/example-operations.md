# Example operations

The three packages under `examples/` are application-level compositions over
the same local engine:

- `examples/directory` opens one complete caller-owned directory. It also works
  with the documented complete Hugging Face snapshot/cache layout after an
  external tool has downloaded and pinned it.
- `examples/embedded` accepts a caller-owned `fs.FS` (including `embed.FS`) and
  an explicit existing WorkDir for retained materialization.
- `examples/adk` borrows one already-open Model and creates application-owned
  in-memory ADK runners for typed data and deterministic routing graphs.

Directory and filesystem examples own their Runtime and Model and always close
Model before Runtime. The ADK example does not close or configure its borrowed
Model. All reports omit raw state and contain only typed decisions, counts,
route names, and immutable runtime/bundle identities.

`task example:contract` builds and tests every example without native inputs.
`task example:audit` adds offline capability, lifecycle, secret, large-file,
and protected-path checks. Protected `task example:gate` and
`task example:gate:race` require the three already-local native/bundle paths,
deny network proxies, and execute directory, fs.FS, accepted-route, fallback,
repeat-use, and coordinated-shutdown paths. The fs.FS case needs one additional
bundle-sized temporary copy; allow about 4 GiB RAM and sufficient WorkDir disk.

For a real compile-time `embed.FS` check (not `os.DirFS`), set the three
protected native/bundle paths from the native runbook and an existing absolute
`LAYA_EMBED_WORKDIR`, then run `task bundle:materialize-embedded` from the repository
root. The probe verifies and copies the complete local bundle into a private
temporary Go module, compiles it with `//go:embed`, compares native directory
and embedded predictions, checks missing/corrupt embedded manifests, and
removes only its own temporary module after the run. It needs space for a
bundle-sized copy and a large Go test binary; it never downloads or commits
weights. Run `task test:embedded:race` to repeat under the race detector. The
`example:gate` operation above tests the generic `fs.FS` path; it does not
substitute for this compile-time embed check.

The examples never download, resolve, authenticate, cache, or prune models.
Those choices, ADK permissions/retries, iteration limits, and downstream route
effects remain application policy. To roll back, remove the example composition
and continue calling the unchanged `laya.Model` API directly; never mutate an
external cache while a Model is open.
