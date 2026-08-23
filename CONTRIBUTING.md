# Contributing

Thank you for helping improve the GMrepo Interactive Portal.

## Before opening a change

- Keep pull requests focused on one concern.
- Do not commit raw participant-level data, credentials, or Streamlit secrets.
- Describe any change to statistical definitions, thresholds, phenotype mapping,
  or taxonomic filtering explicitly.
- If a change alters a generated table, document the input version and regenerate
  all dependent artifacts together.

## Local checks

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
python -m compileall -q GMrepo_Interactive_Portal.py input_files_generation.py
python -m pytest
```

For user-interface changes, also launch the application and inspect all four
pages at desktop width:

```bash
streamlit run GMrepo_Interactive_Portal.py
```

## Reporting problems

Open a GitHub issue with the smallest reproducible example you can provide. For
data or analysis problems, include the affected phenotype, rank, taxon, filters,
and the command used to generate the tables.
