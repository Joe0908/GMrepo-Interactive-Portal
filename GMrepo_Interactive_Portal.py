from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import to_hex, to_rgb

REL_ABUNDANCE_PRESENCE_THRESHOLD = 0.0001
DEFAULT_MIN_VALID_RUNS_FOR_TAXON_PLOTS = 50
APP_TITLE = "GMrepo Interactive Portal"
APP_SUBTITLE = "Fast precomputed version for Taxon Explorer, Phenotype–Taxon Association, and Phenotype Comparisons"
PREVALENCE_START = "#851718"
PREVALENCE_END = "#f5d9cb"
ABUNDANCE_START = "#234086"
ABUNDANCE_END = "#e8eaf6"


@dataclass(frozen=True)
class DataPaths:
    phenotype_taxon_summary: Path
    disease_vs_healthy_comparisons: Path


def _resolve_existing_path(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def get_data_paths() -> DataPaths:
    """
    Resolve data locations in a deployment-friendly order:

    1. Explicit environment variables, if provided.
    2. Files stored in ./data next to this script in the repository.
    3. Files stored next to this script in the repository root.

    This works locally and on Streamlit Community Cloud.
    """
    app_dir = Path(__file__).resolve().parent

    phenotype_env = os.environ.get("GMREPO_PHENOTYPE_TAXON_SUMMARY")
    comparison_env = os.environ.get("GMREPO_DISEASE_COMPARISONS")

    phenotype_candidates = []
    comparison_candidates = []

    if phenotype_env:
        phenotype_candidates.append(Path(phenotype_env).expanduser())
    if comparison_env:
        comparison_candidates.append(Path(comparison_env).expanduser())

    phenotype_candidates.extend(
        [
            app_dir / "data" / "phenotype_taxon_summary.parquet",
            app_dir / "phenotype_taxon_summary.parquet",
        ]
    )
    comparison_candidates.extend(
        [
            app_dir / "data" / "disease_vs_healthy_comparisons.parquet",
            app_dir / "disease_vs_healthy_comparisons.parquet",
        ]
    )

    phenotype_path = _resolve_existing_path(phenotype_candidates)
    comparison_path = _resolve_existing_path(comparison_candidates)

    if phenotype_path is None or comparison_path is None:
        missing = []
        if phenotype_path is None:
            missing.append(
                "phenotype_taxon_summary.parquet "
                "(put it in ./data/ or set GMREPO_PHENOTYPE_TAXON_SUMMARY)"
            )
        if comparison_path is None:
            missing.append(
                "disease_vs_healthy_comparisons.parquet "
                "(put it in ./data/ or set GMREPO_DISEASE_COMPARISONS)"
            )
        raise FileNotFoundError(
            "Required precomputed data file(s) not found:\n- "
            + "\n- ".join(missing)
        )

    return DataPaths(
        phenotype_taxon_summary=phenotype_path,
        disease_vs_healthy_comparisons=comparison_path,
    )


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")


def safe_read_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def sanitize_download_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "figure"


def to_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def generate_value_based_colors(values, start_hex: str, end_hex: str) -> list[str]:
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return []
    vmax = np.nanmax(values)
    if vmax <= 0:
        return [end_hex] * len(values)
    start_rgb = np.array(to_rgb(start_hex))
    end_rgb = np.array(to_rgb(end_hex))
    colors = []
    for v in values:
        t = float(v) / float(vmax)
        rgb = end_rgb + t * (start_rgb - end_rgb)
        colors.append(to_hex(rgb))
    return colors


@st.cache_data(show_spinner=False)
def load_fast_data(paths: DataPaths) -> dict[str, pd.DataFrame]:
    summary = safe_read_table(paths.phenotype_taxon_summary)
    comparisons = safe_read_table(paths.disease_vs_healthy_comparisons)

    for col in ["rank", "taxon", "phenotype"]:
        if col in summary.columns:
            summary[col] = (
                summary[col].astype(str).str.lower().str.strip()
                if col == "rank"
                else summary[col].astype(str).str.strip()
            )

    summary_numeric = [
        "valid_runs",
        "detected_runs",
        "prevalence",
        "prevalence_pct",
        "mean_abundance",
        "mean_abundance_pct",
        "median_abundance",
        "median_abundance_pct",
        "mean_abundance_detected_only",
        "mean_detected_only_abundance_pct",
        "sd_abundance",
    ]
    for col in summary_numeric:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    for col in ["rank", "taxon", "disease", "enriched_in"]:
        if col in comparisons.columns:
            comparisons[col] = (
                comparisons[col].astype(str).str.lower().str.strip()
                if col == "rank"
                else comparisons[col].astype(str).str.strip()
            )

    comparison_numeric = [
        "log2_fc",
        "abs_log2_fc",
        "p",
        "q",
        "combined_prevalence",
        "healthy_prevalence",
        "disease_prevalence",
        "median_healthy",
        "median_disease",
        "mean_detected_healthy",
        "mean_detected_disease",
    ]
    for col in comparison_numeric:
        if col in comparisons.columns:
            comparisons[col] = pd.to_numeric(comparisons[col], errors="coerce")

    if "abs_log2_fc" not in comparisons.columns and "log2_fc" in comparisons.columns:
        comparisons["abs_log2_fc"] = comparisons["log2_fc"].abs()

    if (
        "enriched_in" not in comparisons.columns
        and "log2_fc" in comparisons.columns
        and "disease" in comparisons.columns
    ):
        comparisons["enriched_in"] = np.where(
            comparisons["log2_fc"] > 0,
            comparisons["disease"],
            "Healthy",
        )

    return {"summary": summary, "comparisons": comparisons}


def render_plotly_figure_with_png_download(
    fig,
    filename_base: str,
    chart_key: Optional[str] = None,
    png_scale: int = 3,
):
    fig.update_layout(
        font=dict(family="Arial", size=14),
        title=dict(font=dict(family="Arial", size=24), x=0.5, xanchor="center"),
        margin=dict(l=260, r=100, t=110, b=100),
        coloraxis_colorbar=dict(
            title=dict(font=dict(size=15, family="Arial", color="black")),
            tickfont=dict(size=13, family="Arial", color="black"),
            len=0.7,
            thickness=18,
        ),
    )

    y_title_text = fig.layout.yaxis.title.text if fig.layout.yaxis.title else None
    x_title_text = fig.layout.xaxis.title.text if fig.layout.xaxis.title else None

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        title=dict(
            text=y_title_text,
            font=dict(size=15, color="black", family="Arial"),
            standoff=20,
        ),
        tickfont=dict(size=13, color="black"),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title=dict(
            text=x_title_text,
            font=dict(size=15, color="black", family="Arial"),
            standoff=28,
        ),
        tickfont=dict(size=13, color="black"),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": sanitize_download_name(filename_base),
                "scale": png_scale,
            },
        },
        key=chart_key,
    )


def plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    top_n: int,
    title: str,
    rank_label: Optional[str] = None,
    log_scale: bool = False,
    chart_key: Optional[str] = None,
    y_col: str = "taxon",
):
    if df.empty:
        st.info("No data to display.")
        return

    plot_df = df.head(top_n).copy()
    if metric == "prevalence":
        plot_df["value"] = plot_df["prevalence"] * 100
        x_title = "Prevalence (%)"
        value_fmt = ".2f"
        start_color, end_color = PREVALENCE_START, PREVALENCE_END
    elif metric == "mean_abundance":
        plot_df["value"] = plot_df["mean_abundance"]
        x_title = "Mean relative abundance (%)"
        value_fmt = ".4f"
        start_color, end_color = ABUNDANCE_START, ABUNDANCE_END
    elif metric == "median_abundance":
        plot_df["value"] = plot_df["median_abundance"]
        x_title = "Median relative abundance (%)"
        value_fmt = ".4f"
        start_color, end_color = ABUNDANCE_START, ABUNDANCE_END
    elif metric == "mean_abundance_detected_only":
        plot_df["value"] = plot_df["mean_abundance_detected_only"]
        x_title = "Mean abundance when detected (%)"
        value_fmt = ".4f"
        start_color, end_color = ABUNDANCE_START, ABUNDANCE_END
    else:
        plot_df["value"] = plot_df[metric]
        x_title = metric
        value_fmt = ".2f"
        start_color, end_color = ABUNDANCE_START, ABUNDANCE_END

    y_title = "Phenotype" if str(rank_label).strip().lower() == "phenotype" else "Taxon"

    if log_scale:
        plot_df = plot_df[plot_df["value"] > 0].copy()
    if plot_df.empty:
        st.info("No positive values available for plotting under current settings.")
        return

    plot_df = plot_df.sort_values("value", ascending=False).reset_index(drop=True)
    plot_df["bar_color"] = generate_value_based_colors(
        plot_df["value"], start_color, end_color
    )
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["value"],
            y=plot_df[y_col],
            orientation="h",
            marker=dict(color=plot_df["bar_color"], line=dict(width=0)),
            text=[format(v, value_fmt) for v in plot_df["value"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [
                    plot_df["detected_runs"]
                    if "detected_runs" in plot_df.columns
                    else np.repeat(np.nan, len(plot_df)),
                    plot_df["valid_runs"]
                    if "valid_runs" in plot_df.columns
                    else np.repeat(np.nan, len(plot_df)),
                    plot_df["prevalence"]
                    if "prevalence" in plot_df.columns
                    else np.repeat(np.nan, len(plot_df)),
                    plot_df["mean_abundance"]
                    if "mean_abundance" in plot_df.columns
                    else np.repeat(np.nan, len(plot_df)),
                    plot_df["median_abundance"]
                    if "median_abundance" in plot_df.columns
                    else np.repeat(np.nan, len(plot_df)),
                    plot_df["mean_abundance_detected_only"]
                    if "mean_abundance_detected_only" in plot_df.columns
                    else np.repeat(np.nan, len(plot_df)),
                    plot_df["sd_abundance"]
                    if "sd_abundance" in plot_df.columns
                    else np.repeat(np.nan, len(plot_df)),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{x_title}: %{{x:{value_fmt}}}<br>"
                "Detected runs: %{customdata[0]}<br>"
                "Valid runs: %{customdata[1]}<br>"
                "Prevalence: %{customdata[2]:.4f}<br>"
                "Mean abundance: %{customdata[3]:.6f}<br>"
                "Median abundance: %{customdata[4]:.6f}<br>"
                "Mean abundance when detected: %{customdata[5]:.6f}<br>"
                "SD abundance: %{customdata[6]:.6f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        height=max(500, 26 * min(top_n, len(plot_df)) + 120),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)

    if log_scale:
        fig.update_xaxes(type="log")

    render_plotly_figure_with_png_download(
        fig,
        filename_base=title.lower().replace(" ", "_").replace("–", "-"),
        chart_key=chart_key,
    )


def plot_taxon_log2fc_across_diseases(
    df: pd.DataFrame,
    taxon_name: str,
    rank: str,
    top_n_healthy: int = 10,
    top_n_disease: int = 10,
    chart_key: Optional[str] = None,
):
    if df.empty:
        st.info("No comparison results available for this taxon.")
        return

    plot_df = df.copy()
    plot_df["abs_log2_fc"] = plot_df["log2_fc"].abs()
    healthy_df = plot_df.loc[plot_df["log2_fc"] < 0].copy()
    disease_df = plot_df.loc[plot_df["log2_fc"] > 0].copy()
    healthy_top = (
        healthy_df.sort_values("log2_fc", ascending=True)
        .head(int(top_n_healthy))
        .copy()
        if int(top_n_healthy) > 0
        else healthy_df.iloc[0:0].copy()
    )
    disease_top = (
        disease_df.sort_values("log2_fc", ascending=False)
        .head(int(top_n_disease))
        .copy()
        if int(top_n_disease) > 0
        else disease_df.iloc[0:0].copy()
    )
    plot_df = pd.concat([healthy_top, disease_top], axis=0, ignore_index=True)
    if plot_df.empty:
        st.info("No comparison results available for this taxon.")
        return

    plot_df = plot_df.sort_values("log2_fc", ascending=True)
    max_abs = float(plot_df["abs_log2_fc"].max()) if not plot_df.empty else 1.0
    if max_abs == 0:
        max_abs = 1.0

    fig = px.bar(
        plot_df,
        x="log2_fc",
        y="disease",
        orientation="h",
        color="log2_fc",
        color_continuous_scale="RdBu_r",
        range_color=[-max_abs, max_abs],
        title=f"log2 fold change profile across Healthy vs disease comparisons: {taxon_name} ({rank})",
        hover_data={
            "disease": True,
            "log2_fc": ":.3f",
            "median_healthy": ":.6f",
            "median_disease": ":.6f",
            "mean_detected_healthy": ":.6f",
            "mean_detected_disease": ":.6f",
            "p": ":.3e" if "p" in plot_df.columns else False,
            "q": ":.3e",
            "enriched_in": True,
            "significant": True if "significant" in plot_df.columns else False,
            "abs_log2_fc": False,
        },
        text="log2_fc",
    )
    fig.update_traces(texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
    fig.update_layout(
        height=max(500, 32 * len(plot_df) + 140),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis={"categoryorder": "array", "categoryarray": plot_df["disease"].tolist()},
        coloraxis_colorbar=dict(title="log2 FC"),
        font=dict(family="Arial", size=12, color="#1f355e"),
        title=dict(x=0.5, xanchor="center", font=dict(size=22, family="Arial", color="#1f355e")),
        margin=dict(l=80, r=80, t=90, b=80),
    )
    fig.update_xaxes(
        title="log2 fold change",
        showgrid=True,
        gridcolor="#d5dbe5",
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickfont=dict(size=12, color="#1f355e"),
        title_font=dict(size=16, color="#1f355e"),
    )
    fig.update_yaxes(
        title="Disease",
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickfont=dict(size=12, color="#1f355e"),
        title_font=dict(size=16, color="#1f355e"),
    )
    render_plotly_figure_with_png_download(
        fig,
        filename_base=f"{sanitize_download_name(taxon_name)}_{rank}_log2fc_across_diseases",
        chart_key=chart_key,
    )
    st.caption(
        "Negative scores indicate enrichment in Healthy. Positive scores indicate enrichment in disease. "
        "Within each direction, the strongest comparisons are shown first."
    )


def plot_disease_vs_healthy_log2fc(
    df: pd.DataFrame,
    disease: str,
    rank: str,
    abs_log2fc_cutoff: float = 1.0,
    top_n_healthy: int = 10,
    top_n_disease: int = 10,
    chart_key: Optional[str] = None,
):
    if df.empty:
        st.info("No comparison results available.")
        return

    plot_df = df.copy()
    plot_df["abs_log2_fc"] = plot_df["log2_fc"].abs()
    plot_df = plot_df.loc[plot_df["abs_log2_fc"] >= abs_log2fc_cutoff].copy()
    if plot_df.empty:
        st.info("No taxa passed the current log2 fold change threshold.")
        return

    healthy_df = plot_df.loc[plot_df["log2_fc"] < 0].copy()
    disease_df = plot_df.loc[plot_df["log2_fc"] > 0].copy()
    healthy_top = (
        healthy_df.sort_values("log2_fc", ascending=True)
        .head(int(top_n_healthy))
        .copy()
        if int(top_n_healthy) > 0
        else healthy_df.iloc[0:0].copy()
    )
    disease_top = (
        disease_df.sort_values("log2_fc", ascending=False)
        .head(int(top_n_disease))
        .copy()
        if int(top_n_disease) > 0
        else disease_df.iloc[0:0].copy()
    )
    plot_df = pd.concat([healthy_top, disease_top], axis=0, ignore_index=True)
    if plot_df.empty:
        st.info("No taxa are available to plot after directional filtering.")
        return

    log2fc_values = plot_df["log2_fc"]
    has_positive = (log2fc_values > 0).any()
    has_negative = (log2fc_values < 0).any()
    if has_positive and has_negative:
        max_abs = np.abs(log2fc_values).max()
        zmin, zmax, colorscale = -max_abs, max_abs, "RdBu"
    elif has_positive:
        zmin, zmax, colorscale = 0, log2fc_values.max(), "Reds"
    elif has_negative:
        zmin, zmax, colorscale = log2fc_values.min(), 0, "Blues"
    else:
        zmin, zmax, colorscale = -1, 1, "RdBu"

    plot_df["direction_group"] = np.where(
        plot_df["log2_fc"] > 0,
        f"Enriched in {disease}",
        "Enriched in Healthy",
    )
    plot_df = plot_df.sort_values("log2_fc", ascending=True).copy()
    fig = px.bar(
        plot_df,
        x="log2_fc",
        y="taxon",
        orientation="h",
        color="log2_fc",
        color_continuous_scale=colorscale,
        range_color=[zmin, zmax],
        title=f"Healthy vs {disease}: top enriched taxa by log2 fold change",
        hover_data={
            "taxon": True,
            "log2_fc": ":.3f",
            "abs_log2_fc": ":.3f",
            "median_healthy": ":.6f",
            "median_disease": ":.6f",
            "mean_detected_healthy": ":.6f",
            "mean_detected_disease": ":.6f",
            "p": ":.3e" if "p" in plot_df.columns else False,
            "q": ":.3e",
            "enriched_in": True,
            "significant": True,
            "direction_group": True,
        },
        text="log2_fc",
    )
    fig.update_traces(texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False)
    fig.add_vline(x=0, line_width=1.2, line_dash="dash", line_color="black")
    fig.update_layout(
        height=max(650, 34 * len(plot_df) + 160),
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_colorbar=dict(title="log2 FC"),
        yaxis=dict(categoryorder="array", categoryarray=plot_df["taxon"].tolist()),
        font=dict(family="Arial", size=12, color="#1f355e"),
        title=dict(x=0.5, xanchor="center", font=dict(size=22, family="Arial", color="#1f355e")),
        margin=dict(l=80, r=80, t=90, b=80),
    )
    fig.update_xaxes(
        title="log2 fold change",
        showgrid=True,
        gridcolor="#d5dbe5",
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickfont=dict(size=12, color="#1f355e"),
        title_font=dict(size=16, color="#1f355e"),
    )
    fig.update_yaxes(
        title=f"Taxon ({rank})",
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickfont=dict(size=12, color="#1f355e"),
        title_font=dict(size=16, color="#1f355e"),
    )
    render_plotly_figure_with_png_download(
        fig,
        filename_base=f"healthy_vs_{sanitize_download_name(disease)}_{rank}_top_directional_log2fc",
        chart_key=chart_key,
        png_scale=3,
    )

    if int(top_n_healthy) == 0:
        st.caption(
            f"Only taxa enriched in {disease} are shown. Taxa are ranked by log2 fold change "
            "after applying the selected FDR and effect-size thresholds."
        )
    elif int(top_n_disease) == 0:
        st.caption(
            "Only taxa enriched in Healthy are shown. Taxa are ranked by log2 fold change "
            "after applying the selected FDR and effect-size thresholds."
        )
    else:
        st.caption(
            f"Bars on the left indicate taxa enriched in Healthy, and bars on the right indicate taxa enriched in {disease}. "
            "Within each direction, taxa are ranked by the magnitude of log2 fold change. "
            "If fewer than the requested number are available in either direction, all available taxa are shown."
        )


def page_home(summary_df: pd.DataFrame, comparisons_df: pd.DataFrame):
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    c1, c2, c3 = st.columns(3)
    c1.metric("Phenotypes", f"{summary_df['phenotype'].nunique():,}")
    c2.metric("Taxa in summary", f"{summary_df[['rank', 'taxon']].drop_duplicates().shape[0]:,}")
    c3.metric("Healthy vs disease rows", f"{len(comparisons_df):,}")
    st.markdown(
        "This version is fast because it reads precomputed summary tables instead of "
        "recalculating from the raw run-level abundance file during interaction."
    )


def page_taxon_explorer(summary_df: pd.DataFrame, comparisons_df: pd.DataFrame):
    st.title("Taxon Explorer")

    rank_options = sorted(summary_df["rank"].dropna().unique().tolist())
    default_rank_idx = rank_options.index("genus") if "genus" in rank_options else 0
    rank = st.selectbox("Taxonomic level", rank_options, index=default_rank_idx)
    taxon_options = sorted(
        summary_df.loc[summary_df["rank"] == rank, "taxon"].dropna().unique().tolist()
    )
    taxon = st.selectbox("Select taxon", taxon_options)

    c1, c2, c3 = st.columns(3)
    with c1:
        rank_by = st.selectbox(
            "Rank phenotypes by",
            [
                "prevalence",
                "mean_abundance",
                "median_abundance",
                "mean_abundance_detected_only",
            ],
            index=0,
            format_func=lambda x: {
                "prevalence": "Prevalence",
                "mean_abundance": "Mean abundance",
                "median_abundance": "Median abundance",
                "mean_abundance_detected_only": "Mean abundance when detected",
            }[x],
        )
    with c2:
        top_n = st.selectbox("Top phenotypes to show", [10, 15, 20, 30, 50], index=2)
    with c3:
        min_valid_runs = st.number_input(
            "Minimum valid runs per phenotype",
            min_value=1,
            value=DEFAULT_MIN_VALID_RUNS_FOR_TAXON_PLOTS,
            step=1,
        )

    taxon_summary_df = summary_df[
        (summary_df["rank"] == rank) & (summary_df["taxon"] == taxon)
    ].copy()
    if taxon_summary_df.empty:
        st.info("This taxon was not found in the filtered summary data.")
        return
    taxon_summary_df = taxon_summary_df[
        taxon_summary_df["valid_runs"] >= min_valid_runs
    ].copy()
    if taxon_summary_df.empty:
        st.info("No phenotypes remain after applying the valid-run filter.")
        return

    taxon_summary_df = taxon_summary_df.sort_values(
        by=[rank_by, "prevalence", "mean_abundance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    st.markdown("### Phenotypes ranked for selected taxon")
    title_map = {
        "prevalence": f"Phenotypes ranked by prevalence for {taxon}",
        "mean_abundance": f"Phenotypes ranked by mean abundance for {taxon}",
        "median_abundance": f"Phenotypes ranked by median abundance for {taxon}",
        "mean_abundance_detected_only": f"Phenotypes ranked by mean abundance when detected for {taxon}",
    }
    plot_metric_bar(
        df=taxon_summary_df,
        metric=rank_by,
        top_n=top_n,
        title=title_map[rank_by],
        rank_label="phenotype",
        log_scale=False,
        chart_key=f"taxon_explorer_{sanitize_download_name(taxon)}_{rank}_{rank_by}_{top_n}",
        y_col="phenotype",
    )
    st.caption(
        "Each bar represents one phenotype. Prevalence is the proportion of valid runs "
        "in which the selected taxon is detected above the abundance threshold."
    )

    st.markdown("### Table view")
    display_df = taxon_summary_df[
        [
            "phenotype",
            "detected_runs",
            "valid_runs",
            "prevalence_pct",
            "mean_abundance_pct",
            "median_abundance_pct",
            "mean_detected_only_abundance_pct",
            "sd_abundance",
        ]
    ].rename(
        columns={
            "phenotype": "Phenotype",
            "detected_runs": "Detected runs",
            "valid_runs": "Valid runs",
            "prevalence_pct": "Prevalence (%)",
            "mean_abundance_pct": "Mean abundance (%)",
            "median_abundance_pct": "Median abundance (%)",
            "mean_detected_only_abundance_pct": "Mean abundance when detected (%)",
            "sd_abundance": "SD abundance (%)",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
    st.download_button(
        "Download phenotype table as CSV",
        data=to_download_bytes(display_df),
        file_name=f"taxon_explorer_{sanitize_download_name(taxon)}_{rank}.csv",
        mime="text/csv",
    )




def page_phenotype_taxon_association(summary_df: pd.DataFrame):
    st.title("Phenotype–Taxon Association")
    phenotypes = sorted(summary_df["phenotype"].dropna().unique().tolist())
    phenotype = st.selectbox("Select phenotype", phenotypes)

    c1, c2, c3 = st.columns(3)
    with c1:
        rank_options = sorted(summary_df["rank"].dropna().unique().tolist())
        default_rank_idx = rank_options.index("genus") if "genus" in rank_options else 0
        rank = st.selectbox("Taxonomic level", rank_options, index=default_rank_idx)
    with c2:
        top_n = st.selectbox("Show top N", [15, 20, 50, 100, 200, 500], index=1)
    with c3:
        metric = st.selectbox(
            "Rank taxa by",
            [
                "prevalence",
                "mean_abundance",
                "median_abundance",
                "mean_abundance_detected_only",
            ],
            index=0,
        )

    df = summary_df[
        (summary_df["phenotype"] == phenotype) & (summary_df["rank"] == rank)
    ].copy()
    df = df.sort_values([metric, "taxon"], ascending=[False, True]).head(int(top_n))
    if df.empty:
        st.info("No taxa are available for the selected phenotype and rank.")
        return

    st.markdown("### Taxa ranked for selected phenotype")
    title_map = {
        "prevalence": f"{phenotype}: top taxa by prevalence",
        "mean_abundance": f"{phenotype}: top taxa by mean abundance",
        "median_abundance": f"{phenotype}: top taxa by median abundance",
        "mean_abundance_detected_only": f"{phenotype}: top taxa by mean abundance when detected",
    }
    plot_metric_bar(
        df=df,
        metric=metric,
        top_n=int(top_n),
        title=title_map[metric],
        rank_label=rank,
        chart_key=f"{sanitize_download_name(phenotype)}_{rank}_{metric}",
        y_col="taxon",
    )

    st.markdown("### Table view")
    display_df = df[
        [
            "taxon",
            "detected_runs",
            "valid_runs",
            "prevalence_pct",
            "mean_abundance_pct",
            "median_abundance_pct",
            "mean_detected_only_abundance_pct",
            "sd_abundance",
        ]
    ].rename(
        columns={
            "taxon": "Taxon",
            "detected_runs": "Detected runs",
            "valid_runs": "Valid runs",
            "prevalence_pct": "Prevalence (%)",
            "mean_abundance_pct": "Mean abundance (%)",
            "median_abundance_pct": "Median abundance (%)",
            "mean_detected_only_abundance_pct": "Mean abundance when detected (%)",
            "sd_abundance": "SD abundance (%)",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
    st.download_button(
        "Download phenotype association table as CSV",
        data=to_download_bytes(display_df),
        file_name=f"{sanitize_download_name(phenotype)}_{rank}_association.csv",
        mime="text/csv",
    )



def page_phenotype_comparisons(comparisons_df: pd.DataFrame):
    st.title("Phenotype Comparisons")
    diseases = sorted(comparisons_df["disease"].dropna().unique().tolist())
    ranks = sorted(comparisons_df["rank"].dropna().unique().tolist())
    disease_rank_options = [r for r in ["genus", "species"] if r in ranks] or ranks

    r1c1, r1c2, r1c3, r1c4 = st.columns([2.4, 1.2, 1.2, 1.2])
    r2c1, r2c2 = st.columns([1.2, 1.2])

    with r1c1:
        selected_disease = st.selectbox(
            "Select disease phenotype",
            diseases,
            key="comparison_selected_disease",
        )
    with r1c2:
        default_rank_idx = (
            disease_rank_options.index("genus") if "genus" in disease_rank_options else 0
        )
        selected_rank = st.selectbox(
            "Taxonomic level",
            disease_rank_options,
            index=default_rank_idx,
            key="comparison_rank",
        )
    with r1c3:
        prevalence_threshold = st.number_input(
            "Minimum prevalence",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="comparison_prevalence_threshold",
        )
    with r1c4:
        top_n_disease = st.number_input(
            f"Top {selected_disease}-enriched taxa",
            min_value=1,
            value=20,
            step=1,
            key="top_n_disease_first",
        )
    with r2c1:
        q_threshold = st.selectbox(
            "Maximum FDR (q-value)",
            options=[0.05, 0.01, 0.001],
            index=0,
            key="comparison_q_threshold",
        )
    with r2c2:
        abs_log2fc_cutoff = st.number_input(
            "Minimum absolute log2 fold change",
            min_value=0.5,
            max_value=10.0,
            value=0.5,
            step=0.1,
        )

    top_n_healthy = 0
    df = comparisons_df[
        (comparisons_df["disease"] == selected_disease)
        & (comparisons_df["rank"] == selected_rank)
    ].copy()
    df = df[
        (df["combined_prevalence"] >= prevalence_threshold)
        & (df["q"] <= q_threshold)
    ].copy()
    df["significant"] = (
        (df["q"] <= q_threshold) & (df["log2_fc"].abs() >= abs_log2fc_cutoff)
    )

    if df.empty:
        st.info("No taxa passed the comparison filters.")
        return

    st.markdown("### Comparison results")
    plot_disease_vs_healthy_log2fc(
        df=df,
        disease=selected_disease,
        rank=selected_rank,
        abs_log2fc_cutoff=abs_log2fc_cutoff,
        top_n_healthy=top_n_healthy,
        top_n_disease=top_n_disease,
        chart_key="healthy_vs_disease_log2fc",
    )

    display_df = df.rename(
        columns={
            "taxon": "Taxon",
            "disease": "Disease",
            "median_healthy": "Median abundance in Healthy",
            "median_disease": "Median abundance in disease",
            "log2_fc": "log2 fold change",
            "p": "P-value",
            "q": "FDR (q-value)",
            "enriched_in": "Enriched in",
            "significant": "Significant",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)
    st.download_button(
        "Download comparison table as CSV",
        data=to_download_bytes(display_df),
        file_name=f"phenotype_comparison_healthy_vs_{sanitize_download_name(selected_disease)}.csv",
        mime="text/csv",
    )


def main():
    try:
        paths = get_data_paths()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    data = load_fast_data(paths)
    summary_df = data["summary"]
    comparisons_df = data["comparisons"]

    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to",
            [
                "Home / Overview",
                "Taxon Explorer",
                "Phenotype–Taxon Association",
                "Phenotype Comparisons",
            ],
            index=0,
        )
        st.caption("Data files loaded from the repository or environment variables.")

    if page == "Home / Overview":
        page_home(summary_df, comparisons_df)
    elif page == "Taxon Explorer":
        page_taxon_explorer(summary_df, comparisons_df)
    elif page == "Phenotype–Taxon Association":
        page_phenotype_taxon_association(summary_df)
    elif page == "Phenotype Comparisons":
        page_phenotype_comparisons(comparisons_df)


if __name__ == "__main__":
    main()

