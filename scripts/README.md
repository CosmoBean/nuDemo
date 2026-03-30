# Scripts

The `Makefile` is intentionally thin. Most tunable behavior now lives in small bash wrappers here.

The pattern is:

1. Pick the closest wrapper.
2. Set env vars inline or edit a copied script.
3. Run it directly or through `make`.

Examples:

```bash
LIMIT=4096 SCENE_LIMIT=200 BACKENDS="redis lance parquet" NUM_WORKERS="0 4" \
  bash scripts/run_benchmark.sh real

BACKEND=minio-postgres LIMIT=1024 \
  bash scripts/run_storage.sh

Q=pedestrian SOURCE=elasticsearch \
  bash scripts/run_indexing.sh track-search
```

If you want a machine-specific profile, copy a wrapper into `scripts/custom/` and change the defaults at the top.
