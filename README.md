# GMrepo Interactive Portal

[![CI](https://github.com/Joe0908/GMrepo-Interactive-Portal/actions/workflows/ci.yml/badge.svg)](https://github.com/Joe0908/GMrepo-Interactive-Portal/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B.svg)](https://streamlit.io/)

An interactive Streamlit application for exploring phenotype-level patterns in
human gut microbiome relative-abundance data derived from GMrepo v3.

**[Open the live application](https://gmrepo-interactive-app-rgpoe3s4hmhdbuzdcofnc4.streamlit.app/)**

## What the portal does

The portal separates three questions that are often conflated in microbiome
association analyses:

1. **Taxon Explorer:** in which phenotypes is a selected taxon prevalent or
   abundant?
2. **Phenotype–Taxon Association:** which taxa are most prevalent or abundant
   within a selected phenotype?
3. **Phenotype Comparisons:** which taxa differ between a disease phenotype and
   the pooled healthy group under the implemented filtering and testing rules?

The current bundled release covers 233 phenotype labels and 4,973 unique taxon
labels (2,158 genera and 2,815 species). These counts describe the processed
tables in this repository, not the complete current GMrepo database.

## Repository structure

| Path | Purpose |
|---|---|
| `GMrepo_Interactive_Portal.py` | Streamlit application and visualisation code |
| `input_files_generation.py` | Command-line preprocessing and precomputation pipeline |
| `data/` | Bundled precomputed Parquet tables used by the deployed application |
| `tests/` | Schema and pipeline regression tests |
| `.github/workflows/ci.yml` | Automated lint, compile, and test checks |

## Quick start

Python 3.10 or later is required.

```bash
git clone https://github.com/Joe0908/GMrepo-Interactive-Portal.git
cd GMrepo-Interactive-Portal

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run GMrepo_Interactive_Portal.py
```

The application reads the two bundled files in `data/` by default. Alternative
locations can be supplied with the environment variables
`GMREPO_PHENOTYPE_TAXON_SUMMARY` and `GMREPO_DISEASE_COMPARISONS`.

## Rebuild the precomputed tables

The preprocessing script accepts tab-separated, comma-separated, or Parquet
input tables. It standardises the relevant fields, builds a run-level abundance
table, and writes the two tables consumed by the application.

```bash
python input_files_generation.py \
  --sample-metadata path/to/sample_metadata.tsv \
  --abundance path/to/abundance.tsv \
  --taxonomy path/to/taxonomy.tsv \
  --outdir data_generated
```

The taxonomy table is optional. Run `python input_files_generation.py --help`
for all options. See [`data/README.md`](data/README.md) for the output schemas,
checksums, and current artifact sizes.

## Implemented analysis

- A taxon is counted as detected when its relative abundance is at least
  `0.0001` in the source table's abundance units.
- Prevalence is `detected_runs / valid_runs` within a phenotype.
- Mean and median relative abundance are intended to include non-detections as
  zeros; a detected-only mean is also reported.
- Disease-versus-healthy comparisons use a two-sided Mann–Whitney U test.
- P-values are adjusted by Benjamini–Hochberg within each disease–rank
  comparison.
- Log2 fold change is calculated from group medians with a pseudocount of
  `1e-9`.

## Interpretation and scope

This portal is an exploratory research tool, not a clinical diagnostic system.
The data are compositional, observational, and aggregated across independent
studies. Apparent phenotype associations may reflect cohort composition,
protocol, geography, medication, diet, or other study-level effects. The portal
does not establish causality, disease specificity, or cross-cohort
reproducibility.

## Data source and citation

The processed inputs were derived from [GMrepo](https://gmrepo.humangut.info/).
When using the underlying resource, cite the current GMrepo publication:

> Liu C, Wang X, Zhang Z, et al. GMrepo v3: a curated human gut microbiome
> database with expanded disease coverage and enhanced cross-dataset biomarker
> analysis. *Nucleic Acids Research*. 2026;54(D1):D734–D742.
> [doi:10.1093/nar/gkaf1190](https://doi.org/10.1093/nar/gkaf1190)

The repository does not currently declare a software license. Source-data and
derived-data use remains subject to the terms of the original data provider.

## Development

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
python -m compileall -q GMrepo_Interactive_Portal.py input_files_generation.py
python -m pytest
```

Small, focused contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).
