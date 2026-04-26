# GMrepo Interactive Portal

An interactive web platform for exploring microbiome–phenotype associations using precomputed data derived from the GMrepo v3 database.

## Overview

This project provides a structured analytical framework to investigate three distinct dimensions of microbiome–phenotype relationships:

- Cross-phenotype taxonomic prevalence and abundance
- Within-phenotype taxon dominance
- Disease vs healthy differential enrichment

The platform enables interactive exploration of these analyses through a web interface built with Streamlit.

## Live Application

The interactive web application is available at:

http://gmrepo-interactive-app-rgpoe3s4hmhdbuzdcofnc4.streamlit.app/#g-mrepo-interactive-portal



## Data Processing Pipeline

The script `input_files_generation.py` performs the full data processing workflow:

1. Merge GMrepo metadata, abundance, and taxonomy tables  
2. Standardise phenotype labels and remove low-information taxa  
3. Construct a run-level abundance table  
4. Compute phenotype–taxon summary statistics:
   - prevalence
   - mean and median abundance (zero-filled)
5. Perform disease vs healthy comparisons:
   - Mann–Whitney U test  
   - log2 fold change  
   - Benjamini–Hochberg FDR correction  

The outputs are saved as:

- `phenotype_taxon_summary.parquet`
- `disease_vs_healthy_comparisons.parquet`

## Usage

### Run the pipeline

```bash
python input_files_generation.py \
  --sample-metadata path/to/sample_metadata.tsv \
  --abundance path/to/abundance.tsv \
  --taxonomy path/to/taxonomy.tsv \
  --outdir data
