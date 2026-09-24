# syntax=docker/dockerfile:1

# The published linux/x64 package contains the native binary and pinned
# ONNX Runtime. Model acquisition and bundle conversion remain operator-owned.
FROM node:24-bookworm-slim AS packages
ARG LAYAJEV_VERSION=0.2.6
RUN test "$(dpkg --print-architecture)" = amd64 \
    && npm install --prefix /opt/layajev --omit=dev --ignore-scripts \
      --no-audit --no-fund --no-package-lock \
      "@metalagman/layajev@${LAYAJEV_VERSION}" \
    && test -x /opt/layajev/node_modules/@metalagman/layajev-linux-x64/bin/layajev \
    && test -f /opt/layajev/node_modules/@metalagman/layajev-linux-x64/bin/libonnxruntime.so.1.29.0

FROM ubuntu:24.04
ARG LAYAJEV_VERSION=0.2.6
LABEL org.opencontainers.image.source="https://github.com/metalagman/laya-go" \
      org.opencontainers.image.title="layajev" \
      org.opencontainers.image.version="${LAYAJEV_VERSION}"
RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=packages /opt/layajev/node_modules/@metalagman/layajev-linux-x64 /opt/layajev

USER 65532:65532
RUN /opt/layajev/bin/layajev doctor
EXPOSE 8080/tcp
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl --fail --silent --show-error --max-time 3 http://127.0.0.1:8080/v1/models >/dev/null || exit 1
ENTRYPOINT ["/opt/layajev/bin/layajev", "serve", "--bundle", "/models", "--listen", "0.0.0.0:8080", "--allow-remote"]
