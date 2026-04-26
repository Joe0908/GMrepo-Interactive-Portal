#!/usr/bin/env python3
"""
GMrepo full preprocessing and precomputation pipeline.

This single script converts the three downloaded GMrepo input tables into:
  1. run_level_abundance.tsv
  2. phenotype_taxon_summary.parquet
  3. disease_vs_healthy_comparisons.parquet

Example:
python gmrepo_full_precompute_pipeline.py \
  --sample-metadata data_raw/sample_metadata.tsv \
  --abundance data_raw/abundance.tsv \
  --taxonomy data_raw/taxonomy.tsv \
  --outdir data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

REL_ABUNDANCE_PRESENCE_THRESHOLD = 0.0001
HEALTHY_LABEL = "Healthy"
DEFAULT_RANKS = ["genus", "species"]
EPS = 1e-9

LOW_INFORMATION_TAXON_KEYWORDS = [
    "unclassified",
    "uncultured",
    "unidentified",
    "unknown",
    "metagenome",
    "environmental sample",
]

PHENOTYPE_REPLACEMENTS = {
    "health": "Healthy",
    "healthy": "Healthy",
    "healthy control": "Healthy",
    "control": "Healthy",
    "controls": "Healthy",
    "normal": "Healthy",
    "crohn disease": "Crohn Disease",
    "crohn's disease": "Crohn Disease",
    "crohns disease": "Crohn Disease",
    "ulcerative colitis": "Ulcerative Colitis",
    "parkinson disease": "Parkinson Disease",
    "parkinson's disease": "Parkinson Disease",
    "non alcoholic fatty liver disease": "Non-alcoholic Fatty Liver Disease",
    "non-alcoholic fatty liver disease": "Non-alcoholic Fatty Liver Disease",
    "type 1 diabetes": "Type 1 Diabetes",
    "type 2 diabetes": "Type 2 Diabetes",
    "covid 19": "COVID-19",
    "covid-19": "COVID-19",
}


def safe_read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "none": np.nan, "": np.nan})
    )


def first_present(df: pd.DataFrame, candidate_groups: list[list[str]]) -> Optional[str]:
    cols_lower = {str(col).strip().lower(): col for col in df.columns}
    for group in candidate_groups:
        for candidate in group:
            matched = cols_lower.get(candidate.lower())
            if matched is not None:
                return matched
    return None


def canonicalize_phenotype(value: object) -> object:
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value == "":
        return np.nan
    return PHENOTYPE_REPLACEMENTS.get(value.lower(), value)


def is_low_information_taxon(name: object) -> bool:
    if pd.isna(name):
        return True
    name_str = str(name).strip().lower()
    if name_str == "":
        return True
    return any(keyword in name_str for keyword in LOW_INFORMATION_TAXON_KEYWORDS)


def rank_priority(rank_value: object) -> int:
    if pd.isna(rank_value):
        return 0
    priority_map = {
        "species": 5,
        "subspecies": 4,
        "strain": 3,
        "genus": 2,
        "family": 1,
    }
    return priority_map.get(str(rank_value).strip().lower(), 0)


def standardize_sample_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    df = meta.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = normalize_text_series(df[col])

    run_col = first_present(df, [["run_id", "run", "accession_id"], ["sample_id", "second_sample_id"]])
    phenotype_col = first_present(df, [["phenotype"], ["disease"], ["term"]])
    project_col = first_present(df, [["project_id", "bioproject"]])
    experiment_col = first_present(df, [["experiment_type", "data_type", "seq_type"]])

    if run_col is None:
        raise ValueError("sample metadata must contain a run ID column")
    if phenotype_col is None:
        raise ValueError("sample metadata must contain phenotype, disease, or term")

    rename_map = {run_col: "run_id", phenotype_col: "phenotype"}
    if project_col is not None:
        rename_map[project_col] = "project_id"
    if experiment_col is not None:
        rename_map[experiment_col] = "experiment_type"
    df = df.rename(columns=rename_map)

    if "project_id" not in df.columns:
        df["project_id"] = np.nan
    if "experiment_type" not in df.columns:
        df["experiment_type"] = np.nan

    df["run_id"] = normalize_text_series(df["run_id"])
    df["phenotype"] = df["phenotype"].map(canonicalize_phenotype)
    df["project_id"] = normalize_text_series(df["project_id"])
    df["experiment_type"] = normalize_text_series(df["experiment_type"])

    before = len(df)
    df = df.dropna(subset=["run_id", "phenotype"]).drop_duplicates(subset=["run_id"]).copy()
    print(f"Sample metadata: {before:,} rows -> {len(df):,} retained")
    return df[["run_id", "phenotype", "project_id", "experiment_type"]]


def standardize_taxonomy(tax: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if tax is None:
        return None
    df = tax.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = normalize_text_series(df[col])

    taxid_col = first_present(df, [["ncbi_taxon_id"], ["taxon_id"], ["id"]])
    name_col = first_present(df, [["scientific_name"], ["name"]])
    rank_col = first_present(df, [["node_rank"], ["rank"], ["taxon_rank_level"]])
    if taxid_col is None or name_col is None:
        raise ValueError("taxonomy must contain ncbi_taxon_id/taxon_id and scientific_name/name")

    rename_map = {taxid_col: "ncbi_taxon_id", name_col: "scientific_name"}
    if rank_col is not None:
        rename_map[rank_col] = "node_rank"
    df = df.rename(columns=rename_map)
    if "node_rank" not in df.columns:
        df["node_rank"] = np.nan

    df["ncbi_taxon_id"] = df["ncbi_taxon_id"].astype(str).str.strip()
    df["scientific_name"] = normalize_text_series(df["scientific_name"])
    df["node_rank"] = normalize_text_series(df["node_rank"]).astype("string").str.lower().str.strip()

    before = len(df)
    df = df.dropna(subset=["ncbi_taxon_id", "scientific_name"]).copy()
    df["rank_priority"] = df["node_rank"].map(rank_priority)
    df = df.sort_values(["ncbi_taxon_id", "rank_priority", "scientific_name"], ascending=[True, False, True], kind="stable")
    df = df.drop_duplicates(subset=["ncbi_taxon_id"], keep="first").drop(columns=["rank_priority"])
    print(f"Taxonomy: {before:,} rows -> {len(df):,} retained")
    return df[["ncbi_taxon_id", "scientific_name", "node_rank"]]


def standardize_abundance_long(abund: pd.DataFrame) -> pd.DataFrame:
    df = abund.copy()
    run_col = first_present(df, [["accession_id"], ["run_id", "run"], ["loaded_uid"]])
    taxid_col = first_present(df, [["ncbi_taxon_id"], ["taxon_id"]])
    abundance_col = first_present(df, [["relative_abundance"], ["abundance"], ["rel_abundance"], ["value"]])
    rank_col = first_present(df, [["taxon_rank_level"], ["rank"], ["node_rank"]])

    if run_col is None or taxid_col is None or abundance_col is None:
        raise ValueError("abundance must contain run/accession ID, taxon ID, and relative abundance columns")

    rename_map = {run_col: "run_id", taxid_col: "ncbi_taxon_id", abundance_col: "relative_abundance"}
    if rank_col is not None:
        rename_map[rank_col] = "rank"
    df = df.rename(columns=rename_map)
    if "rank" not in df.columns:
        df["rank"] = np.nan

    df["run_id"] = normalize_text_series(df["run_id"])
    df["ncbi_taxon_id"] = df["ncbi_taxon_id"].astype(str).str.strip()
    df["rank"] = normalize_text_series(df["rank"]).astype("string").str.lower().str.strip()
    df["relative_abundance"] = pd.to_numeric(df["relative_abundance"], errors="coerce").fillna(0.0)

    before = len(df)
    df = df.dropna(subset=["run_id", "ncbi_taxon_id"]).copy()
    print(f"Abundance: {before:,} rows -> {len(df):,} retained")
    return df[["run_id", "ncbi_taxon_id", "rank", "relative_abundance"]]


def build_run_level_abundance(sample_meta: pd.DataFrame, abundance: pd.DataFrame, taxonomy: Optional[pd.DataFrame], drop_low_information_taxa: bool = True) -> pd.DataFrame:
    df = abundance.copy()
    if taxonomy is not None:
        df = df.merge(
            taxonomy[["ncbi_taxon_id", "scientific_name", "node_rank"]].rename(columns={"node_rank": "taxonomy_rank"}),
            on="ncbi_taxon_id",
            how="left",
        )
    else:
        df["scientific_name"] = df["ncbi_taxon_id"]
        df["taxonomy_rank"] = np.nan

    df["scientific_name"] = normalize_text_series(df["scientific_name"])
    df["taxonomy_rank"] = normalize_text_series(df["taxonomy_rank"]).astype("string").str.lower().str.strip()
    df["final_rank"] = df["taxonomy_rank"]

    df = df.merge(sample_meta, on="run_id", how="left")
    before_join_filter = len(df)
    df = df.dropna(subset=["phenotype"]).copy()
    print(f"Joined run-level table: {before_join_filter:,} rows -> {len(df):,} with phenotype")

    if drop_low_information_taxa:
        before = len(df)
        df = df.loc[~df["scientific_name"].map(is_low_information_taxon)].copy()
        print(f"Low-information taxon filter: {before:,} rows -> {len(df):,} retained")

    output_cols = [
        "run_id", "phenotype", "project_id", "experiment_type", "ncbi_taxon_id",
        "scientific_name", "rank", "taxonomy_rank", "final_rank", "relative_abundance",
    ]
    for col in output_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[output_cols].copy()


def prepare_run_level_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"run_id", "phenotype", "relative_abundance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input run-level table missing required columns: {sorted(missing)}")

    out = df.copy()
    out["run_id"] = normalize_text_series(out["run_id"])
    out["phenotype"] = normalize_text_series(out["phenotype"])

    if "scientific_name" not in out.columns:
        raise ValueError("Input run-level table must contain scientific_name")
    out["scientific_name"] = normalize_text_series(out["scientific_name"])

    if "final_rank" not in out.columns:
        if "rank" not in out.columns:
            raise ValueError("Input run-level table must contain final_rank or rank")
        out["final_rank"] = out["rank"]

    out["final_rank"] = normalize_text_series(out["final_rank"]).str.lower()
    out["relative_abundance"] = pd.to_numeric(out["relative_abundance"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["run_id", "phenotype", "scientific_name", "final_rank"]).copy()
    out = out[out["relative_abundance"] >= 0].copy()
    return out


def build_phenotype_taxon_summary(run_level_df: pd.DataFrame) -> pd.DataFrame:
    df = run_level_df[["phenotype", "run_id", "final_rank", "scientific_name", "relative_abundance"]].copy()

    valid_runs_df = (
        df[["phenotype", "run_id"]]
        .drop_duplicates()
        .groupby("phenotype", as_index=False)["run_id"]
        .nunique()
        .rename(columns={"run_id": "valid_runs"})
    )

    agg_obs = (
        df.groupby(["phenotype", "final_rank", "scientific_name"], dropna=False)["relative_abundance"]
        .agg(sum_abundance="sum", sd_abundance="std")
        .reset_index()
    )
    agg_obs["sd_abundance"] = agg_obs["sd_abundance"].fillna(0.0)

    detected = (
        df.loc[df["relative_abundance"] >= REL_ABUNDANCE_PRESENCE_THRESHOLD]
        .groupby(["phenotype", "final_rank", "scientific_name"], dropna=False)["run_id"]
        .nunique()
        .reset_index()
        .rename(columns={"run_id": "detected_runs"})
    )

    summary = agg_obs.merge(detected, on=["phenotype", "final_rank", "scientific_name"], how="left")
    summary["detected_runs"] = summary["detected_runs"].fillna(0).astype(int)
    summary = summary.merge(valid_runs_df, on="phenotype", how="left")

    summary["prevalence"] = summary["detected_runs"] / summary["valid_runs"]
    summary["mean_abundance"] = summary["sum_abundance"] / summary["valid_runs"]
    summary["median_abundance"] = 0.0

    need_real_median = summary["detected_runs"] > (summary["valid_runs"] / 2)
    if need_real_median.any():
        targets = summary.loc[need_real_median, ["phenotype", "final_rank", "scientific_name"]].drop_duplicates()
        df_sub = df.merge(targets, on=["phenotype", "final_rank", "scientific_name"], how="inner")
        phen_runs = df[["phenotype", "run_id"]].drop_duplicates()
        full = phen_runs.merge(targets, on="phenotype", how="inner")
        obs = (
            df_sub.groupby(["phenotype", "run_id", "final_rank", "scientific_name"], as_index=False, dropna=False)["relative_abundance"]
            .sum()
        )
        full = full.merge(obs, on=["phenotype", "run_id", "final_rank", "scientific_name"], how="left")
        full["relative_abundance"] = full["relative_abundance"].fillna(0.0)
        med = (
            full.groupby(["phenotype", "final_rank", "scientific_name"], dropna=False)["relative_abundance"]
            .median()
            .reset_index()
            .rename(columns={"relative_abundance": "median_abundance_true"})
        )
        summary = summary.merge(med, on=["phenotype", "final_rank", "scientific_name"], how="left")
        summary.loc[need_real_median, "median_abundance"] = summary.loc[need_real_median, "median_abundance_true"]
        summary = summary.drop(columns=["median_abundance_true"])

    detected_only = (
        df.loc[df["relative_abundance"] >= REL_ABUNDANCE_PRESENCE_THRESHOLD]
        .groupby(["phenotype", "final_rank", "scientific_name"], dropna=False)["relative_abundance"]
        .mean()
        .reset_index()
        .rename(columns={"relative_abundance": "mean_abundance_detected_only"})
    )
    summary = summary.merge(detected_only, on=["phenotype", "final_rank", "scientific_name"], how="left")
    summary["mean_abundance_detected_only"] = summary["mean_abundance_detected_only"].fillna(0.0)

    summary = summary.rename(columns={"final_rank": "rank", "scientific_name": "taxon"})
    summary["prevalence_pct"] = summary["prevalence"] * 100.0
    summary["mean_abundance_pct"] = summary["mean_abundance"]
    summary["median_abundance_pct"] = summary["median_abundance"]
    summary["mean_detected_only_abundance_pct"] = summary["mean_abundance_detected_only"]

    preferred_order = [
        "phenotype", "rank", "taxon", "valid_runs", "detected_runs", "prevalence", "prevalence_pct",
        "mean_abundance", "mean_abundance_pct", "median_abundance", "median_abundance_pct",
        "sd_abundance", "mean_abundance_detected_only", "mean_detected_only_abundance_pct",
    ]
    return summary[preferred_order].sort_values(["phenotype", "rank", "taxon"]).reset_index(drop=True)


def _build_run_taxon_matrix(rank_df: pd.DataFrame, phenotypes: Iterable[str]) -> pd.DataFrame:
    subset = rank_df[rank_df["phenotype"].isin(list(phenotypes))].copy()
    if subset.empty:
        return pd.DataFrame()
    all_runs = subset[["run_id", "phenotype"]].drop_duplicates().copy()
    taxon_obs = subset.groupby(["run_id", "phenotype", "scientific_name"], as_index=False)["relative_abundance"].sum()
    wide = taxon_obs.pivot(index=["run_id", "phenotype"], columns="scientific_name", values="relative_abundance")
    wide = wide.fillna(0.0).reset_index()
    return all_runs.merge(wide, on=["run_id", "phenotype"], how="left").fillna(0.0)


def build_all_disease_vs_healthy_comparisons(run_level_df: pd.DataFrame, ranks: list[str]) -> pd.DataFrame:
    df = run_level_df.copy()
    df["final_rank"] = df["final_rank"].astype(str).str.lower().str.strip()
    df["scientific_name"] = df["scientific_name"].astype(str).str.strip()

    phenotype_order = sorted(df["phenotype"].dropna().astype(str).str.strip().unique().tolist())
    diseases = [p for p in phenotype_order if p != HEALTHY_LABEL]
    if HEALTHY_LABEL not in phenotype_order:
        raise ValueError("Healthy phenotype not found in input data")

    all_results = []
    for rank in ranks:
        rank_df = df[df["final_rank"] == rank].copy()
        if rank_df.empty:
            continue

        valid_runs_map = rank_df[["phenotype", "run_id"]].drop_duplicates().groupby("phenotype")["run_id"].nunique().to_dict()

        for disease in diseases:
            pair_df = rank_df[rank_df["phenotype"].isin([HEALTHY_LABEL, disease])].copy()
            if pair_df.empty:
                continue
            wide = _build_run_taxon_matrix(pair_df, [HEALTHY_LABEL, disease])
            if wide.empty:
                continue
            healthy_df = wide[wide["phenotype"] == HEALTHY_LABEL]
            disease_df = wide[wide["phenotype"] == disease]
            if healthy_df.empty or disease_df.empty:
                continue

            taxa = [c for c in wide.columns if c not in ["run_id", "phenotype"]]
            rows = []
            for taxon in taxa:
                x = pd.to_numeric(healthy_df[taxon], errors="coerce").fillna(0.0)
                y = pd.to_numeric(disease_df[taxon], errors="coerce").fillna(0.0)
                combined = pd.concat([x, y], ignore_index=True)

                try:
                    _, p = mannwhitneyu(x, y, alternative="two-sided")
                except Exception:
                    p = np.nan

                median_healthy = float(np.median(x))
                median_disease = float(np.median(y))
                rows.append({
                    "disease": disease,
                    "rank": rank,
                    "taxon": taxon,
                    "healthy_valid_runs": int(valid_runs_map.get(HEALTHY_LABEL, len(healthy_df))),
                    "disease_valid_runs": int(valid_runs_map.get(disease, len(disease_df))),
                    "combined_prevalence": float((combined >= REL_ABUNDANCE_PRESENCE_THRESHOLD).mean()),
                    "healthy_prevalence": float((x >= REL_ABUNDANCE_PRESENCE_THRESHOLD).mean()),
                    "disease_prevalence": float((y >= REL_ABUNDANCE_PRESENCE_THRESHOLD).mean()),
                    "median_healthy": median_healthy,
                    "median_disease": median_disease,
                    "mean_detected_healthy": float(x[x >= REL_ABUNDANCE_PRESENCE_THRESHOLD].mean()) if (x >= REL_ABUNDANCE_PRESENCE_THRESHOLD).any() else 0.0,
                    "mean_detected_disease": float(y[y >= REL_ABUNDANCE_PRESENCE_THRESHOLD].mean()) if (y >= REL_ABUNDANCE_PRESENCE_THRESHOLD).any() else 0.0,
                    "log2_fc": float(np.log2(median_disease + EPS) - np.log2(median_healthy + EPS)),
                    "p": p,
                })

            comp = pd.DataFrame(rows)
            if comp.empty:
                continue
            valid_p_mask = comp["p"].notna()
            comp["q"] = np.nan
            if valid_p_mask.any():
                comp.loc[valid_p_mask, "q"] = multipletests(comp.loc[valid_p_mask, "p"], method="fdr_bh")[1]
            comp["enriched_in"] = np.where(comp["log2_fc"] > 0, disease, HEALTHY_LABEL)
            comp["abs_log2_fc"] = comp["log2_fc"].abs()
            comp = comp.sort_values(["q", "abs_log2_fc"], ascending=[True, False], na_position="last")
            all_results.append(comp)

    if not all_results:
        return pd.DataFrame(columns=[
            "disease", "rank", "taxon", "healthy_valid_runs", "disease_valid_runs", "combined_prevalence",
            "healthy_prevalence", "disease_prevalence", "median_healthy", "median_disease",
            "mean_detected_healthy", "mean_detected_disease", "log2_fc", "p", "q", "enriched_in", "abs_log2_fc",
        ])
    return pd.concat(all_results, ignore_index=True)


def write_outputs(run_level_df: pd.DataFrame, summary_df: pd.DataFrame, comparisons_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    run_level_df.to_csv(outdir / "run_level_abundance.tsv", sep="\t", index=False)
    summary_df.to_parquet(outdir / "phenotype_taxon_summary.parquet", index=False)
    summary_df.to_csv(outdir / "phenotype_taxon_summary.tsv", sep="\t", index=False)
    comparisons_df.to_parquet(outdir / "disease_vs_healthy_comparisons.parquet", index=False)
    comparisons_df.to_csv(outdir / "disease_vs_healthy_comparisons.tsv", sep="\t", index=False)

    manifest = pd.DataFrame([
        {"file": "run_level_abundance.tsv", "rows": len(run_level_df)},
        {"file": "phenotype_taxon_summary.parquet", "rows": len(summary_df)},
        {"file": "disease_vs_healthy_comparisons.parquet", "rows": len(comparisons_df)},
    ])
    manifest.to_csv(outdir / "precompute_manifest.tsv", sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GMrepo run-level and precomputed tables for the Streamlit portal.")
    parser.add_argument("--sample-metadata", required=True, help="Path to GMrepo sample metadata table")
    parser.add_argument("--abundance", required=True, help="Path to GMrepo abundance table")
    parser.add_argument("--taxonomy", required=False, default=None, help="Path to GMrepo taxonomy table")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--ranks", nargs="+", default=DEFAULT_RANKS, help="Ranks to include in disease-vs-Healthy comparisons")
    parser.add_argument("--keep-low-information-taxa", action="store_true", help="Do not remove unclassified/uncultured/unknown taxa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    print("Loading raw GMrepo input tables...")
    sample_meta_raw = safe_read_table(args.sample_metadata)
    abundance_raw = safe_read_table(args.abundance)
    taxonomy_raw = safe_read_table(args.taxonomy) if args.taxonomy else None

    print("Standardizing input tables...")
    sample_meta = standardize_sample_metadata(sample_meta_raw)
    abundance = standardize_abundance_long(abundance_raw)
    taxonomy = standardize_taxonomy(taxonomy_raw)

    print("Building run-level abundance table...")
    run_level_df = build_run_level_abundance(
        sample_meta=sample_meta,
        abundance=abundance,
        taxonomy=taxonomy,
        drop_low_information_taxa=not args.keep_low_information_taxa,
    )
    run_level_df = prepare_run_level_df(run_level_df)
    print(f"Run-level table: {len(run_level_df):,} rows")

    print("Building phenotype_taxon_summary.parquet...")
    summary_df = build_phenotype_taxon_summary(run_level_df)
    print(f"Phenotype-taxon summary: {len(summary_df):,} rows")

    print("Building disease_vs_healthy_comparisons.parquet...")
    ranks = [str(r).lower().strip() for r in args.ranks]
    comparisons_df = build_all_disease_vs_healthy_comparisons(run_level_df, ranks)
    print(f"Disease-vs-Healthy comparisons: {len(comparisons_df):,} rows")

    print(f"Writing outputs to {outdir}...")
    write_outputs(run_level_df, summary_df, comparisons_df, outdir)
    print("Done.")


if __name__ == "__main__":
    main()
