# `layajev` container operations

The root [Dockerfile](../Dockerfile) builds a `linux/amd64` image from the exact
published `@metalagman/layajev@0.2.6` npm release. It includes the native
binary and pinned ONNX Runtime shared library, but **not** model weights,
source snapshots, Go, Node, Python, or a model downloader at runtime. Supply
an already-converted, verified FP32 bundle as a read-only mount. The currently
qualified bundle occupies about 1.3 GiB; allow memory for the model and its
inference workload. Other architectures and precisions are not qualified.

## Build and check

On a `linux/amd64` Docker host, from this checkout:

```sh
docker build --platform linux/amd64 -t layajev:0.2.6 .
docker run --rm --entrypoint /opt/layajev/bin/layajev layajev:0.2.6 doctor
```

The equivalent repository operations are `task container:build` and
`task container:doctor`.

The build needs network access to Docker Hub, Ubuntu package archives, and
npm. The runtime stage uses Ubuntu 24.04 because the published native binary
requires glibc 2.39; Debian bookworm's older glibc cannot run it. The final
stage runs `doctor` as an unprivileged user, so an image build fails if the
packaged native library cannot initialize. `doctor` does not check a model bundle or
HTTP serving. To prepare a bundle, use the [source-checkout runbook](layajev-runbook.md)
or the [published-package from-zero script](layajev-npm-release.md); neither
path runs inside the production container. Never mount an unverified source
snapshot in place of a converted bundle.

## Run behind a reverse proxy

Set `LAYA_BUNDLE_DIR` to an absolute path containing a verified bundle. The
image runs as UID/GID `65532:65532` by default. The example below runs as the
bundle owner's UID/GID so a private `0700` bundle is readable without making
model files world-readable:

```sh
export LAYA_BUNDLE_DIR=/absolute/path/to/verified-bundle
docker run --rm --name layajev \
  --user "$(id -u):$(id -g)" \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  --mount "type=bind,src=$LAYA_BUNDLE_DIR,dst=/models,readonly" \
  --publish 127.0.0.1:8080:8080 --stop-timeout 60 \
  layajev:0.2.6
```

With Go Task installed, `LAYA_BUNDLE_DIR=/absolute/path/to/verified-bundle
task container:serve` applies the same local-port, read-only mount, and
current-UID settings.

The container listens on `0.0.0.0:8080` **inside** its network namespace so
Docker can forward the port. The example publishes it only on the host's
loopback interface. `layajev` has no TLS or authentication; put an
authenticating TLS reverse proxy in front of it and restrict network access.
Do not publish the port on a public host interface without that protection.
The process sets `ORT_DISABLE_TELEMETRY=1` before opening the native runtime;
operators do not need to set it. It neither fetches nor converts the model.

Readiness is `GET /v1/models`; the Docker healthcheck queries it after a
120-second startup grace period. Check the container and make one local query:

```sh
docker inspect --format '{{.State.Health.Status}}' layajev
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
curl --fail --silent --show-error http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"laya-multilingual","state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}'
```

See the [API contract and error handling](layajev-runbook.md#compatibility-and-errors)
before integrating a client. The API is a documented Jev-compatible subset,
not an identical Jev model.

## Operate and recover

The bind-mounted bundle must stay immutable for the entire container lifetime.
Keep it outside the container layer, back it up and roll it forward by
deploying a new immutable directory. Do not update files in place. Mount
read-only, keep the root filesystem read-only, and size the `/tmp` tmpfs and
memory limits for your workload; the example's tmpfs size is a starting point,
not a capacity guarantee. Container logs go to stdout/stderr and should be
collected by the deployment platform. Stop with `docker stop layajev`; the
process drains HTTP and closes the model and runtime during shutdown.

If startup fails, inspect `docker logs layajev`, verify the bundle path and
permissions for the chosen UID, then run the bundle verifier from the
checkout (`LAYA_BUNDLE_DIR=... task bundle:verify`). A failed healthcheck may
mean slow model startup or a failed inference initialization; it is not a
license to expose the API without TLS/auth. Roll back by redeploying the
previous image tag **and** its previously verified bundle. The image does not
own bundle cleanup or external caches.
