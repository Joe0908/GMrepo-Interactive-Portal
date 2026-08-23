# Precomputed data artifacts

The two Parquet files in this directory are derived tables bundled so that the
Streamlit application can run without downloading or processing the full GMrepo
input tables.

| File | Rows × columns | Size | SHA-256 |
|---|---:|---:|---|
| `phenotype_taxon_summary.parquet` | 97,002 × 14 | 5.1 MB | `4275bc7a5eff1ce334ee6275b87d8f4639e559a377da2c338c235e942cac4e6c` |
| `disease_vs_healthy_comparisons.parquet` | 749,253 × 17 | 10.3 MB | `d3f1d70d6e2a3936f9c604db7d8905a23b8a8ef267f53a30e1b90ee06aa196cd` |

## `phenotype_taxon_summary.parquet`

One row represents a phenotype–taxonomic-rank–taxon combination.

| Column | Meaning |
|---|---|
| `phenotype` | Standardised phenotype label |
| `rank` | Taxonomic rank (`genus` or `species` in the bundled file) |
| `taxon` | Scientific taxon label |
| `valid_runs` | Runs available for the phenotype |
| `detected_runs` | Runs meeting the presence threshold for the taxon |
| `prevalence` | `detected_runs / valid_runs` |
| `prevalence_pct` | Prevalence expressed as a percentage |
| `mean_abundance` | Zero-filled mean relative abundance |
| `mean_abundance_pct` | Display-oriented copy of mean abundance |
| `median_abundance` | Zero-filled median relative abundance |
| `median_abundance_pct` | Display-oriented copy of median abundance |
| `sd_abundance` | Standard deviation recorded by the preprocessing pipeline |
| `mean_abundance_detected_only` | Mean among detected runs only |
| `mean_detected_only_abundance_pct` | Display-oriented copy of detected-only mean |

## `disease_vs_healthy_comparisons.parquet`

One row represents a disease–rank–taxon comparison against the pooled healthy
group.

| Column group | Columns |
|---|---|
| Identifiers | `disease`, `rank`, `taxon` |
| Group sizes | `healthy_valid_runs`, `disease_valid_runs` |
| Prevalence | `combined_prevalence`, `healthy_prevalence`, `disease_prevalence` |
| Abundance summaries | `median_healthy`, `median_disease`, `mean_detected_healthy`, `mean_detected_disease` |
| Statistical results | `log2_fc`, `p`, `q`, `enriched_in`, `abs_log2_fc` |

## Regeneration

Generate fresh artifacts with `input_files_generation.py`; do not hand-edit
binary outputs. The raw GMrepo input tables are not included in this repository.
Record the GMrepo release or download date and the exact source files whenever
the artifacts are regenerated.

The repository does not currently declare a license. The bundled derived tables
remain subject to applicable terms from the original data provider.
