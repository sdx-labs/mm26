"""Schema and freshness validation for ingested data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig, _json_default, _read_parquet, _required_kaggle_schemas

import polars as pl


def run_validations(config: PipelineConfig, ingest_manifest: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {"checks": [], "failed": False, "data_dir": str(config.data_dir)}
    files = ingest_manifest["kaggle"]["files"]
    required = _required_kaggle_schemas()

    for dataset_name, required_columns in required.items():
        entry = files.get(dataset_name)
        if entry is None:
            results["checks"].append({"dataset": dataset_name, "ok": False, "error": "missing dataset"})
            results["failed"] = True
            continue

        found_columns = set(entry["columns"])
        missing = sorted(set(required_columns) - found_columns)
        ok = not missing and entry["rows"] > 0
        results["checks"].append(
            {
                "dataset": dataset_name,
                "ok": ok,
                "missing_columns": missing,
                "rows": entry["rows"],
            }
        )
        if not ok:
            results["failed"] = True

    sample = _read_parquet(Path(files["SampleSubmissionStage2"]["artifact"]))
    id_parts = sample.with_columns(
        pl.col("ID").str.split("_").list.get(0).cast(pl.Int64).alias("Season")
    )
    latest_season = int(id_parts["Season"].max())
    freshness_ok = latest_season >= config.target_season
    results["checks"].append(
        {
            "dataset": "SampleSubmissionStage2",
            "ok": freshness_ok,
            "latest_season": latest_season,
            "expected_target": config.target_season,
        }
    )
    if not freshness_ok:
        results["failed"] = True

    results["cbbd_manifest_shape"] = sorted(ingest_manifest["cbbd"]["datasets"].keys())
    if config.mode == "daily":
        cbbd_datasets = ingest_manifest["cbbd"]["datasets"]
        games_ok = cbbd_datasets.get("games", {}).get("status") == "ok" and cbbd_datasets.get("games", {}).get("rows", 0) > 0
        lines_ok = cbbd_datasets.get("lines", {}).get("status") == "ok" and cbbd_datasets.get("lines", {}).get("rows", 0) > 0
        results["checks"].append(
            {
                "dataset": "CBBD_daily_games_lines",
                "ok": bool(games_ok and lines_ok),
                "games_rows": cbbd_datasets.get("games", {}).get("rows", 0),
                "lines_rows": cbbd_datasets.get("lines", {}).get("rows", 0),
            }
        )
    report_path = config.reports_dir / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")

    if results["failed"]:
        raise ValueError(f"Validation failed. See {report_path}")

    return results
