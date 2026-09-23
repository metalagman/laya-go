# Repository instructions

## Scope

- Keep `laya` usable as a library; do not add process ownership, logging
  globals, or command-line policy to the root package.
- Model acquisition is caller-owned. Do not add Hugging Face clients,
  authentication, repository resolution, downloads, or external cache
  management.
- Production inference is local and in-process. Do not add an HTTP, gRPC,
  Python, or child-process inference path.
- Do not modify `pack/callee/**` as part of Prism lifecycle work.

## Go conventions

- Follow Google Go style and standard `gofmt` formatting.
- Prefer concrete return types and narrow consumer-owned interfaces only when
  multiple implementations or testing require them.
- Put `context.Context` first on blocking operations and never store it.
- Preserve wrapped causes with `%w` and support `errors.Is` or `errors.As` for
  caller-actionable categories.
- Document ownership, concurrency, cancellation, and cleanup for exported
  stateful APIs.
- Use table-driven standard-library tests with useful failure messages; avoid
  assertion frameworks and timing sleeps.

## Verification

- Use `Taskfile.yml` as the executable operations interface once present.
- Run formatting, module verification, unit tests, race tests, lint, and
  vulnerability checks appropriate to the changed scope.
- Do not claim support for a platform, model, precision, or native behavior
  without its required integration evidence.
