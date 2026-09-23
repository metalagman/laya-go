# v0.1 release preparation

Version `0.1.0` is an unreleased candidate for
`github.com/metalagman/laya-go`. Source commits and review branches may be
published, but a tag, GitHub Release, or package publication requires the
applicable release gates and review to pass.

Run `task release:sbom` to print the exact Go module inventory (not a formal
SBOM), `task package:verify` to audit committed repository contents, and `task
release:contract` for artifact-independent checks. The complete `task
release:verify` additionally requires the pinned local native libraries and
bundle, an existing absolute `LAYA_EMBED_WORKDIR` with bundle-sized free space,
and a separate immutable previous bundle at `LAYA_ROLLBACK_BUNDLE_DIR`. It runs
native, source, ADK, example, real `embed.FS`, qualification, and rollback
checks. The tracked-source preflight deliberately fails while the implementation
remains uncommitted; a passing local worktree test is not clean-checkout
evidence. `rollback:verify` authenticates the prior bundle and contracts
without changing deployment state; application rollback still needs its own
operational rehearsal.

For the first `v0.1.0` release, there is no prior released bundle. Run
`task release:verify-initial` with the same pinned native inputs and embedded
work directory. It runs every release gate above except `rollback:verify`,
requires the exact `0.1.0` version and no rollback-bundle setting, and ends
with an explicit no-predecessor notice. It is not evidence of rollback and
must not be reused for a later release. If the initial release is unsuitable,
withdraw its recommendation and publish a corrected new version; do not
pretend that the current bundle is a previous version.

The safe setup checks are `task setup:platform` and `task setup:native`; they
install or download nothing. `task bundle:verify` checks a supplied bundle,
`task bundle:materialize-embedded` performs the opt-in full compile-time
embedded/native probe in temporary work storage, `task smoke:offline` exercises
all primary offline inference paths, and `task diagnose:native` reports local
toolchain/artifact identities. `task profile:native` writes CPU and heap
profiles and their test binary only to three distinct new absolute paths
(`LAYA_CPU_PROFILE`, `LAYA_HEAP_PROFILE`, and `LAYA_PROFILE_BINARY`); protect
those files as potentially sensitive process artifacts.
`task package:linux-amd64` creates a deterministic gzip source
archive from committed HEAD at a new absolute `LAYA_PACKAGE_OUTPUT` path and
prints its checksum. Native libraries and weights remain separate reviewed
artifacts, not contents of that source archive. A failed packaging attempt may
leave a clearly named `.tmp.tar` or `.tmp.tar.gz` beside the chosen output for
manual inspection; it never replaces an existing archive.

Before publication, review `CHANGELOG.md`, `SECURITY.md`, LICENSE, module and
native licenses/notices, provenance digests, the linux/amd64-only qualification
boundary, checksums, CI least privilege, and a clean Git diff. Tags and released
artifacts are immutable. Fix a released defect with a new SemVer tag/advisory;
never replace an existing tag or silently broaden support claims.

The release contains build-time Python export tooling but no production Python,
HTTP/gRPC inference service, downloader, provider registry, remote fallback,
credentials, model weights, native library, or external cache content.
