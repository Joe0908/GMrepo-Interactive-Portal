import pandas as pd
import pytest

from input_files_generation import (
    build_all_disease_vs_healthy_comparisons,
    build_phenotype_taxon_summary,
    canonicalize_phenotype,
)


def _synthetic_run_level_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("h1", "Healthy", "genus", "Taxon A", 1.0),
            ("h2", "Healthy", "genus", "Taxon A", 2.0),
            ("h3", "Healthy", "genus", "Taxon A", 0.0),
            ("d1", "Disease X", "genus", "Taxon A", 4.0),
            ("d2", "Disease X", "genus", "Taxon A", 5.0),
            ("d3", "Disease X", "genus", "Taxon A", 6.0),
        ],
        columns=["run_id", "phenotype", "final_rank", "scientific_name", "relative_abundance"],
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("healthy control", "Healthy"),
        ("Crohn's disease", "Crohn Disease"),
        ("covid 19", "COVID-19"),
        ("Unmapped phenotype", "Unmapped phenotype"),
    ],
)
def test_canonicalize_phenotype(raw: str, expected: str) -> None:
    assert canonicalize_phenotype(raw) == expected


def test_summary_statistics_on_synthetic_data() -> None:
    summary = build_phenotype_taxon_summary(_synthetic_run_level_data())
    healthy = summary.loc[summary["phenotype"] == "Healthy"].iloc[0]

    assert healthy["valid_runs"] == 3
    assert healthy["detected_runs"] == 2
    assert healthy["prevalence"] == pytest.approx(2 / 3)
    assert healthy["mean_abundance"] == pytest.approx(1.0)
    assert healthy["median_abundance"] == pytest.approx(1.0)


def test_disease_comparison_on_synthetic_data() -> None:
    comparisons = build_all_disease_vs_healthy_comparisons(
        _synthetic_run_level_data(), ranks=["genus"]
    )

    assert len(comparisons) == 1
    result = comparisons.iloc[0]
    assert result["disease"] == "Disease X"
    assert result["taxon"] == "Taxon A"
    assert result["log2_fc"] > 0
    assert result["enriched_in"] == "Disease X"
    assert 0 <= result["p"] <= 1
    assert 0 <= result["q"] <= 1
