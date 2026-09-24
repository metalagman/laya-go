# syntax=docker/dockerfile:1

# The published linux/x64 package contains the native binary and pinned
# ONNX Runtime. Bundle construction is an explicit, verified image-build step;
# the final image needs no model mount or network access for inference.
FROM node:24-bookworm-slim AS packages
RUN test "$(dpkg --print-architecture)" = amd64 \
    && npm install --prefix /opt/layajev --omit=dev --ignore-scripts \
      --no-audit --no-fund --no-package-lock \
      "@metalagman/layajev@0.2.6" \
    && test -x /opt/layajev/node_modules/@metalagman/layajev-linux-x64/bin/layajev \
    && test -f /opt/layajev/node_modules/@metalagman/layajev-linux-x64/bin/libonnxruntime.so.1.29.0

FROM node:24-trixie-slim AS bundle
RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl git tar util-linux \
    && rm -rf /var/lib/apt/lists/*
ENV UV_NO_PROGRESS=1 npm_config_progress=false
COPY scripts/layajev-from-zero.sh /usr/local/bin/layajev-from-zero
RUN bash /usr/local/bin/layajev-from-zero --bundle-only /build

FROM ubuntu:24.04
LABEL org.opencontainers.image.source="https://github.com/metalagman/laya-go" \
      org.opencontainers.image.title="layajev" \
      org.opencontainers.image.version="0.2.6"
RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=packages /opt/layajev/node_modules/@metalagman/layajev-linux-x64 /opt/layajev
COPY --from=bundle --chown=65532:65532 /build/bundle /models

USER 65532:65532
RUN test -f /models/manifest.json && /opt/layajev/bin/layajev doctor
EXPOSE 8080/tcp
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl --fail --silent --show-error --max-time 3 http://127.0.0.1:8080/v1/models >/dev/null || exit 1
ENTRYPOINT ["/opt/layajev/bin/layajev", "serve", "--bundle", "/models", "--listen", "0.0.0.0:8080", "--allow-remote"]
