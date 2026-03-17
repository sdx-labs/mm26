# MM26 Stage 1 Pipeline

Stage 1 provides a reproducible Kaggle March Machine Learning Mania 2026 pipeline with bronze/silver/gold layers and a submission artifact.

## Run

From repository root:

```bash
python main.py run --mode manual --target-season 2026
```

## Outputs

Each run writes to:

- `artifacts/runs/<run_id>/bronze`
- `artifacts/runs/<run_id>/silver`
- `artifacts/runs/<run_id>/gold`
- `artifacts/runs/<run_id>/reports`
- `artifacts/runs/<run_id>/submission.csv`
- `artifacts/runs/<run_id>/run_manifest.json`

`artifacts/latest` points to the newest run.

## Notebook Contract

Notebooks should read from `artifacts/latest/gold/*` and `artifacts/latest/reports/*` and should not mutate pipeline state.

## Daily Refresh

A scheduled/manual workflow exists in `.github/workflows/daily_pipeline.yml`.
