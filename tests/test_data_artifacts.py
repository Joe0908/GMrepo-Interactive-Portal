from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_summary_artifact_schema() -> None:
    summary = pd.read_parquet(DATA_DIR / "phenotype_taxon_summary.parquet")
    required = {
        "phenotype",
        "rank",
        "taxon",
        "valid_runs",
        "detected_runs",
        "prevalence",
        "mean_abundance",
        "median_abundance",
    }

    assert not summary.empty
    assert required.issubset(summary.columns)
    assert summary["prevalence"].between(0, 1).all()
    assert (summary["detected_runs"] <= summary["valid_runs"]).all()


def test_comparison_artifact_schema() -> None:
    comparisons = pd.read_parquet(DATA_DIR / "disease_vs_healthy_comparisons.parquet")
    required = {
        "disease",
        "rank",
        "taxon",
        "log2_fc",
        "p",
        "q",
        "enriched_in",
    }

    assert not comparisons.empty
    assert required.issubset(comparisons.columns)
    assert comparisons["p"].dropna().between(0, 1).all()
    assert comparisons["q"].dropna().between(0, 1).all()
