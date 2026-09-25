# Shared Qdrant infrastructure

The service configuration is [compose.yaml](compose.yaml). From the repository root:

```text
docker compose -f infrastructure/qdrant/compose.yaml config --quiet
docker compose -f infrastructure/qdrant/compose.yaml up -d
docker compose -f infrastructure/qdrant/compose.yaml ps
```

The image remains `qdrant/qdrant:v1.19.1`. HTTP/dashboard binds to
`127.0.0.1:6333`, and gRPC binds to `127.0.0.1:6334`.

## Preserved storage

The configuration explicitly retains project name `ir-qdrant-learning`, service
`qdrant`, network `ir-qdrant-learning_default`, and Docker volume
`ir-qdrant-learning-data`, mounted at `/qdrant/storage`.
There is no explicit container_name; Compose generates
`ir-qdrant-learning-qdrant-1`. There are no host bind mounts or relative storage
paths. On the existing Docker Desktop host the volume mountpoint is
`/var/lib/docker/volumes/ir-qdrant-learning-data/_data`, inside Docker's Linux
storage.

The Compose file moved from the former learning-demo directory with its contents
unchanged. No extra project flag, service stop, data move, or migration is needed.
Do not override the project or volume names when using this existing installation.
Read-only checks for an existing installation:

```text
docker volume inspect ir-qdrant-learning-data
docker inspect ir-qdrant-learning-qdrant-1
docker compose -f infrastructure/qdrant/compose.yaml ps
```

Do not remove the named volume: it contains completed indexes and potentially
incomplete interrupted MPNet indexes. Resume reuses completed ranking artifacts;
it does not inspect or repair Qdrant. Collection lifecycle stays in the vector
retrieval scripts.

The educational demo source now lives in [theory/qdrant-demo/](../../theory/qdrant-demo/README.md).
The legacy `qdrant-experiment/` directory remains locally only because its
`.venv/` and `model-cache/` must stay in place. Its ignore rules are retained;
these local files are not infrastructure or part of the experiment pipeline.
