# `layajev` npm release and `npx` operations

The omnidist profile stages two MIT-licensed npm packages, `@metalagman/layajev`
and its optional platform package `@metalagman/layajev-linux-x64`. Only
`linux/amd64` with the pinned native build has been qualified. There is no
Darwin, Windows, ARM, musl, or general Linux compatibility claim. Neither npm
package contains model weights, ONNX Runtime, or a Python inference service.
The 1.3 GiB FP32 bundle and the pinned native runtime remain
deployment-owned inputs. The present workflow only builds a candidate; it does
not publish either package.

## Run a published package

After a release has actually appeared on npm, use its exact version on a
compatible `linux/amd64` host:

```sh
export LAYAJEV_VERSION=0.2.0 # replace with the version actually published
export LAYA_BUNDLE_DIR=/absolute/path/to/verified-bundle
export LAYA_ONNXRUNTIME_LIBRARY=/absolute/path/to/libonnxruntime.so.1.29.0
export LAYA_TOKENIZERS_LIBRARY=/absolute/path/to/libtokenizers.a
export ORT_DISABLE_TELEMETRY=1
npx -y "@metalagman/layajev@$LAYAJEV_VERSION" serve --bundle "$LAYA_BUNDLE_DIR"
```

The tokenizer archive is linked into the packaged binary at build time, but
the runtime still requires the reviewed `LAYA_TOKENIZERS_LIBRARY` path. The
ONNX Runtime shared library, local bundle, and telemetry setting are also
required. `CGO_LDFLAGS` is only a build-time setting, not a consumer setting.
See [native dependencies](https://github.com/metalagman/laya-go/blob/main/docs/native-dependencies.md) for pinned library identities
and [source-checkout operations](https://github.com/metalagman/laya-go/blob/main/docs/layajev-runbook.md) for the Jev-compatible
subset, HTTP examples, and safe binding rules. The default server listens on
`127.0.0.1:8080`; it provides neither TLS nor authentication. Do not expose it
directly on a public interface.

The packaged `fetch` and `convert` subcommands are **not** standalone model
preparation tools yet: `fetch` locates a checked-in export profile and
`convert` invokes the repository Taskfile and locked exporter. Run those from
an appropriate source checkout with its prerequisites, or supply an already
verified bundle. Do not expect `npx` alone to acquire or convert a model.

## Prepare a release candidate

From a reviewed `main` checkout, choose a new SemVer version; do not reuse the
existing `v0.1.0` source-release tag or an npm version already published. The
operator must confirm control of the `@metalagman` scope and both package names
before any future publication. Review `.omnidist/omnidist.yaml`, the MIT
license, the pinned native dependency hashes, and the candidate diff. Run the
repository's `task check` plus protected native qualification with the
required external artifacts before making support claims.

The trusted manual workflow `.github/workflows/omnidist-release.yml` accepts a
version on `main`, acquires only the checksum-pinned build-time tokenizer
archive, and performs `build`, `stage`, and `verify`. It uploads staged npm
directories for seven days. It has read-only repository permission and no npm
publish token. A maintainer may start it with:

```sh
gh workflow run omnidist-release.yml --ref main -f version=0.2.0-rc.0
```

For local reproduction on `linux/amd64`, obtain the same verified
`libtokenizers.a` described in [native dependencies](https://github.com/metalagman/laya-go/blob/main/docs/native-dependencies.md)
without changing the source tree, then run:

```sh
export OMNIDIST_VERSION=0.2.0-rc.0
export CGO_LDFLAGS=-L/absolute/path/containing/libtokenizers.a
npx -y @omnidist/omnidist@latest build
npx -y @omnidist/omnidist@latest stage
npx -y @omnidist/omnidist@latest verify
sha256sum .omnidist/default/dist/linux/amd64/layajev \
  .omnidist/default/npm/@metalagman/layajev-linux-x64/bin/layajev
```

Inspect both staged `package.json` files for the same new version, MIT license,
correct names, and a single `linux/x64` optional dependency. Inspect the
staged file list for absence of weights and unintended binaries. Run the
staged binary's `--help`; for a native smoke, use the pinned external runtime
and bundle and query `/v1/models` as described above. The `npx` CLI version
used by the manual workflow is pinned to `@omnidist/omnidist@0.1.37`; local
reproduction with `@latest` must be reviewed if that version has moved.

## Future publication and recovery

Publication is a separate, explicit operator decision. It requires a reviewed
candidate, successful quality/native gates, verified npm account and scope
rights, and a new version. Never put an npm token in Git, workflow YAML, or a
command argument. The profile uses token authentication; provide
`NPM_PUBLISH_TOKEN` privately in a controlled environment. First run
`npx -y @omnidist/omnidist@latest publish --dry-run` on the exact staged
candidate, then use `npx -y @omnidist/omnidist@latest npm publish` only when
the publication decision is made. No automatic tag push or GitHub Release is
configured here.

Npm publication is not atomic: omnidist sends the platform package before the
meta package. If an upload fails, stop retries, preserve the log, and inspect
both `@metalagman/layajev-linux-x64` and `@metalagman/layajev` at that exact
version. Compare accepted files and checksums against the verified staging;
never rebuild different bytes under the same version or move an existing tag.
Resume only the missing units after checking registry state, or issue a new
version and explain the incomplete one. Finally, verify installation and an
external-bundle smoke through the published `npx` command on `linux/amd64`.
