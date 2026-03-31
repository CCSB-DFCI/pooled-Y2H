"""
TODO:

- dot plot of reads
- replicate scatter
- replicates error bar
- combined error bar
- some comparison (pool sizes but can do other?)
- variant success rate (change name of x axis)
- extract n replicates from df

"""

import itertools
from pathlib import Path
import math

import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.transforms as transforms
from matplotlib.patches import Patch
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from utils import _get_n_replicates, sort_nt_var, sort_var
from config import MIN_READS_IN_LW


COLOR_MAP_FOR_CLIN_SIG = {
    "Pathogenic": "#CA7682",
    "Likely pathogenic": "#E6B1B8",
    "Benign": "#1D7AAB",
    "Likely benign": "#63A1C4",
    "Conflicting": "#505050",
    "VUS": "#A0A0A0",
    "Other": "#E0E0E0",
}
COLOR_BENIGN = COLOR_MAP_FOR_CLIN_SIG["Benign"]
COLOR_PATHOGENIC = COLOR_MAP_FOR_CLIN_SIG["Pathogenic"]


EMPTY_AD_ID = 0


def square_subplots(n, inches_per_subplot=2, sharexy=True):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * inches_per_subplot, nrows * inches_per_subplot),
        sharex=sharexy,
        sharey=sharexy,
        squeeze=False,
    )
    for ax in axes.flatten()[n:]:
        ax.axis("off")
    return fig, axes


def plot_growth_scores_by_gene_pies(tb, output_dir, close_fig=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = square_subplots(len(tb[["symbol", "pool_id"]].drop_duplicates()))
    for (gene, i_pool), ax in zip(
        tb[["symbol", "pool_id"]]
        .drop_duplicates()
        .sort_values(["symbol", "pool_id"])
        .values,
        axs.flatten(),
    ):
        vals = tb.loc[
            (tb["symbol"] == gene)
            & (tb["pool_id"] == i_pool)
            & (tb["media"] == "3AT")
            & (tb["interactor_id"] != 0),
            "y2h_score",
        ].value_counts(dropna=False)
        if len(vals) == 0:
            continue  # HACK because of N/A
        for i in list(range(5)) + [np.nan]:
            if i not in vals:
                vals[i] = 0
        colors = {
            0: "black",
            1: "red",
            2: "yellow",
            3: "green",
            4: "blue",
            np.nan: "grey",
        }
        ax.pie(
            vals.sort_index(),
            colors=colors.values(),
            radius=np.sqrt(vals.sum() / 3) * 0.3,
        )
        title = f"{gene}"
        if tb.loc[tb["symbol"] == gene, "pool_id"].nunique() > 1:
            title += f" pool {i_pool}"
        title += f" ({vals.sum() // 3})"
        ax.set_title(title)

    legend_elements = [Patch(facecolor=c, label=str(i)) for i, c in colors.items()]
    axs[-1, -1].legend(handles=legend_elements, title="Growth scores")
    axs[-1, -1].set_title("(# PPIs)")
    for loc in ["top", "bottom", "right", "left"]:
        axs[-1, -1].spines[loc].set_visible(False)

    fig.savefig(output_dir / "PPI-scores_by_gene_pie.pdf", bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_filtering_summary(log, output_dir):
    """
    TODO:
        - check if there's any difference between gene-pool and gene
          and if not, just plot the gene level

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = list(log.values())[0].keys()
    for level in levels:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(w=6, h=0.5 + len(log) * 0.5)
        ax.barh(y=list(log.keys()), width=[v[level] for v in log.values()])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title(level)
        ax.set_xticks([])
        ax.yaxis.set_tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i, v in enumerate(log.values()):
            ax.text(
                x=0,
                y=i,
                s=f"   {v[level]:,} ({v[level] / list(log.values())[0][level]:.1%})",
                ha="left",
                va="center",
                color="white",
                fontsize=10,
            )
        fig.savefig(
            output_dir / f"filtering_summary_{level.replace(' ', '_')}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_read_counts_for_single_gene_pool(df, gene_name, pool_id, close_fig=False):
    """
    TODO:
    - just say pool 1 if there are multiple pools
    - could take None for pool_id arg and check if mutliple pools

    """
    df = (
        df[(df["symbol"] == gene_name) & (df["pool_id"] == pool_id)]
        .copy()
        .sort_values("interactor_symbol")
    )
    if df.empty:
        return
    n_interactors = df["interactor_symbol"].nunique()
    n_rep = _get_n_replicates(df)
    fig, axs = plt.subplots(
        nrows=2, ncols=n_rep, sharex=True, constrained_layout=True, squeeze=False
    )
    fig.set_size_inches(w=n_interactors * 2 * (n_rep / 3) + 2, h=8)
    for media, ax_row in zip(["lw", "3at"], axs):
        for i_rep, ax in zip(range(1, n_rep + 1), ax_row):
            sns.stripplot(
                data=df.sort_values(["interactor_symbol"]),
                y=f"read_cnt_{media}{i_rep}",
                x="interactor_symbol",
                alpha=0.5,
                ax=ax,
            )
            ax.set_ylim(0, max(ax.get_ylim()[1], 10))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    for i_rep, ax in zip(range(1, 4), axs[0, :]):
        ax.set_title(f"Replicate {i_rep}")
    axs[0, min(1, axs.shape[1] - 1)].set_title(
        f"$\\bf{{{{{gene_name}\\;\\;Pool\\,{pool_id}}}}}$\nReplicate 2"
    )
    for ax in axs[1, :]:
        ax.set_xlabel("")
        ax.xaxis.set_tick_params(rotation=90)
    for ax in axs[:, 1:].flatten():
        ax.set_ylabel("")
    axs[0, 0].set_ylabel("Read Count -LW")
    axs[1, 0].set_ylabel("Read Count -LWH +3AT")
    fig.savefig(
        f"../output/figures/{df['experiment'].iloc[0]}/per_gene/{gene_name}_pool_{pool_id}_read_count_per_variant.pdf",
        bbox_inches="tight",
    )
    if close_fig:
        plt.close(fig)


def plot_pct_per_variant(
    df,
    gene_name,
    i_pool,
    media="LW",
    y_max=None,
    threshold=0.1,
    min_abs_count=MIN_READS_IN_LW,
    doubled_wt=True,
    draw_stats=True,
    close_fig=False,
    out_path=None,
):
    """ """
    if media not in ("LW", "3AT", "QC"):
        raise ValueError(
            f"Invalid media type: {media}. Expected one of: 'LW', '3AT', 'QC'"
        )
    if media == "LW":
        media_label = "-LW"
        media_col = "lw"
    elif media == "3AT":
        media_label = "-LWH +3AT"
        media_col = "3at"
    elif media == "QC":
        media_label = "QC"
        media_col = "qc"
    df = (
        df.loc[(df["symbol"] == gene_name) & (df["pool_id"] == i_pool), :]
        .copy()
        .sort_values(
            ["interactor_symbol", "var_id"],
            key=lambda x: x.apply(sort_var) if x.name == "var_id" else x,
        )
    )
    if df.empty:
        raise ValueError(f"No data available for {gene_name} in pool {i_pool}")

    n_reps = _get_n_replicates(df)
    n_cols = n_reps
    n_ints = df["interactor_symbol"].nunique()
    n_var = df.loc[df["pool_id"] == i_pool, "var_id"].nunique()
    fig, axs = plt.subplots(
        ncols=n_cols,
        nrows=n_ints,
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    fig.set_size_inches(w=n_cols * (n_var * 0.2 + 1), h=n_ints * 2.5)
    for i_int, ax_row in zip(df["interactor_symbol"].unique(), axs):
        for i_rep in range(1, n_reps + 1):
            ax = ax_row[i_rep - 1]
            counts = df.loc[
                (df["interactor_symbol"] == i_int), f"read_cnt_{media_col}{i_rep}"
            ].values
            if counts.sum() == 0:
                # invisible dummy bar to get constrained layout to work
                ax.bar(
                    x=df.loc[df["pool_id"] == i_pool, "var_id"].unique(),
                    height=[10] * df.loc[df["pool_id"] == i_pool, "var_id"].nunique(),
                    alpha=0,
                )
                continue
            pct = counts / counts.sum() * 100
            ax.bar(
                x=df.loc[df["pool_id"] == i_pool, "var_id"].unique(),
                height=pct,
            )
            exp_pct = 100 / (n_var + 1) if doubled_wt == True else 100 / n_var
            min_abs_count_as_pct = min_abs_count / counts.sum() * 100
            ax.axhline(y=exp_pct, color="grey", lw=1)
            if media == "LW":
                ax.axhline(
                    y=max([exp_pct * threshold, min_abs_count_as_pct]),
                    color="red",
                    lw=1,
                )

            if draw_stats:
                # drop WT and filter out low abundance variants
                pct_filtered = pct[1:][
                    (pct[1:] >= (exp_pct * threshold)) & (counts[1:] >= min_abs_count)
                ]
                drop_out = np.mean(
                    (pct < (exp_pct * threshold)) | (counts < min_abs_count)
                )
                if len(pct_filtered) < 2:
                    std = np.nan
                    mad = np.nan
                else:
                    std = np.std(pct_filtered)
                    mad = stats.median_abs_deviation(pct_filtered)
                transform_x_data_y_ax = transforms.blended_transform_factory(
                    ax.transData, ax.transAxes
                )
                ax.text(
                    s=f"Drop out = {drop_out:.0%}\nSTD = {std:.1f}%\nMAD = {mad:.1f}%",
                    x=n_var - 0.5,
                    y=0.95,
                    transform=transform_x_data_y_ax,
                    fontsize=7,
                    ha="right",
                    va="top",
                )

    for ax in axs.flatten():
        ax.set_ylim(0, y_max)
    for i_int, ax_row in zip(df["interactor_symbol"].unique(), axs):
        for i_rep in range(1, n_reps + 1):
            ax = ax_row[i_rep - 1]
            counts = df.loc[
                (df["interactor_symbol"] == i_int), f"read_cnt_{media_col}{i_rep}"
            ].values
            if counts.sum() == 0:
                continue
            ax_count = ax.twinx()
            y_pct_max = ax.get_ylim()[1]
            ax_count.set_ylim(0, counts.sum() * (y_pct_max / 100))
            if i_rep == 3:
                ax_count.set_ylabel("Read count")
            ax_count.yaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{x:,.0f}")
            )
    for ax in axs[-1, :]:
        ax.xaxis.set_tick_params(rotation=90)
    for interactor_name, ax in zip(df["interactor_symbol"].unique(), axs[:, 0]):
        ax.set_ylabel(f"$\\mathbf{{{interactor_name}}}$\nFraction of reads (%)")

    middle_rep = list(range(1, n_reps + 1))[int(n_reps / 2)]
    for i_rep in range(1, n_reps + 1):
        title = "Replicate " + str(i_rep)
        if i_rep == middle_rep:
            title = (
                f"$\\mathbf{{{gene_name}\\;\\;pool\\:{i_pool}\\;\\;{media_label}}}$\n\n"
                + title
            )
        axs[0, i_rep - 1].set_title(title)

    if out_path is None:
        out_dir = Path(f"../output/figures/{df['experiment'].values[0]}/per_gene")
        out_path = (
            out_dir
            / f"{gene_name}_pool_{i_pool}_{media}-fraction-per-variant-interactor-replicate_bars.pdf"
        )
    else:
        out_path = Path(out_path)
        out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_log2fc_replicate_scatter(
    df,
    description="overall",
    alt_ref=None,
    min_val=-10,
    max_val=6,
    alpha=0.05,
    close_fig=False,
):
    """
    - Add growth score
    - implement alt_ref
    """
    if df.empty:
        return
    df = df.copy()
    n_replicates = _get_n_replicates(df)
    fig, axs = plt.subplots(
        nrows=1, ncols=n_replicates, sharex=True, sharey=True, squeeze=False
    )
    ax_row = axs[0, :]
    fig.set_size_inches(w=n_replicates * 3, h=3)
    _plot_log2fc_replicate_scatter_row(
        df,
        ax_row,
        alt_ref=alt_ref,
        min_val=min_val,
        max_val=max_val,
        alpha=alpha,
    )
    plt.subplots_adjust(hspace=0.4)
    fig.savefig(
        f"../output/figures/{df['experiment'].iloc[0]}/{description}_log2fc-by-replicates_scatter-with-errors.pdf",
        bbox_inches="tight",
    )
    if close_fig:
        plt.close(fig)


def plot_log2fc_replicate_scatter_per_gene_pool(
    df,
    gene,
    pool_id,
    alt_ref=None,
    min_val=-10,
    max_val=6,
    alpha=0.9,
    close_fig=False,
    description="",
):
    """
    - Add growth score if column in df
    - implement alt_ref
    - check for out of bounds
    """
    df = df.loc[(df["symbol"] == gene) & (df["pool_id"] == pool_id), :].copy()
    if df.empty:
        return
    n_replicates = _get_n_replicates(df)
    interactors = df["interactor_symbol"].sort_values().unique()
    n_interactors = len(interactors)
    fig, axs = plt.subplots(
        nrows=n_interactors,
        ncols=n_replicates,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.set_size_inches(w=n_replicates * 3, h=n_interactors * 3)
    for i_interactor, ax_row in zip(interactors, axs):
        df_f = df.loc[df["interactor_symbol"] == i_interactor, :]
        _plot_log2fc_replicate_scatter_row(
            df_f,
            ax_row,
            alt_ref=alt_ref,
            min_val=min_val,
            max_val=max_val,
            alpha=alpha,
        )
        ax_row[min(1, axs.shape[1] - 1)].set_title(f"{i_interactor}")
    axs[0, min(1, axs.shape[1] - 1)].set_title(
        f"$\\bf{{{{{gene}\\;\\;Pool {pool_id}}}}}$\n{interactors[0]}"
    )

    plt.subplots_adjust(hspace=0.4)
    fig.savefig(
        f"../output/figures/{df['experiment'].iloc[0]}/per_gene/{gene}_pool_{pool_id}_log2fc-by-replicates_scatter-with-errors{description}.pdf",
        bbox_inches="tight",
    )
    if close_fig:
        plt.close(fig)


def _plot_log2fc_replicate_scatter_row(
    df, ax_row, alt_ref=None, min_val=-10, max_val=6, alpha=0.9
):
    df = df.copy()
    n_replicates = _get_n_replicates(df)
    for (j, k), ax in zip(
        itertools.combinations(range(1, n_replicates + 1), 2), ax_row
    ):
        ax.plot([min_val, max_val], [min_val, max_val], color="grey", linewidth=1)
        ax.axhline(y=0, color="grey", linestyle="-", linewidth=1)
        ax.axvline(x=0, color="grey", linestyle="-", linewidth=1)
        x = df[f"log2FC_{j}"].values
        y = df[f"log2FC_{k}"].values
        if np.all(np.isnan(x)) or np.all(np.isnan(y)):
            continue
        ax.errorbar(
            x=x,
            xerr=df[f"error_log2FC_{j}"].values,
            y=y,
            yerr=df[f"error_log2FC_{k}"].values,
            fmt="o",
            alpha=alpha,
            markersize=2,
        )
        ax.set_ylim(min_val, max_val)
        ax.set_xlim(min_val, max_val)
        ax.set_xlabel(f"Log2FC - Replicate {j}")
        ax.set_ylabel(f"Log2FC - Replicate {k}")
        if (~np.isnan(x) & ~np.isnan(y)).sum() < 2:
            continue  # can plot just one point but can't calculate a PCC
        valid_points = (~np.isnan(x) & ~np.isnan(y)).copy()
        x = x[valid_points]
        y = y[valid_points]
        r = stats.pearsonr(x, y)[0]
        sigma_x = df[f"error_log2FC_{j}"].values
        sigma_y = df[f"error_log2FC_{k}"].values
        sigma_x = sigma_x[valid_points]
        sigma_y = sigma_y[valid_points]
        chi2 = np.sum(((y - x) / np.sqrt(sigma_x**2 + sigma_y**2)) ** 2)
        chi2_r = chi2 / len(x)
        ax.text(
            x=0.05,
            y=0.97,
            s=f"R² = {r**2:.2f}\nχ²ᵣ = {chi2_r:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )


def _plot_log2fc_with_error_for_interactor(
    df,
    interactor,
    variants,
    rep="combined",
    ax=None,
    color_map=COLOR_MAP_FOR_CLIN_SIG,
    **kwargs,
):
    log2fc = []
    err = []
    for v in variants:
        # TODO: check only one value
        log2fc.append(
            df.loc[
                (df["interactor_symbol"] == interactor) & (df["var_id"] == v),
                f"log2FC_{rep}",
            ].values[0]
        )
        err.append(
            df.loc[
                (df["interactor_symbol"] == interactor) & (df["var_id"] == v),
                f"error_log2FC_{rep}",
            ].values[0]
        )
    ax.errorbar(
        y=variants, x=log2fc, xerr=err, fmt="o", capsize=5, clip_on=False, **kwargs
    )
    yticklabels = ax.get_yticklabels()
    if "cv_clinical_significance_clean" in df.columns:
        for label, v in zip(yticklabels, variants):
            sig = df.loc[df["var_id"] == v, "cv_clinical_significance_clean"].iloc[0]
            if sig is not None:
                label.set_color(color_map.get(sig, "black"))


def plot_pooled_Y2H_results(
    df,
    gene_name,
    i_pool,
    plot_combined=True,
    plot_replicates=False,
    include_empty_AD=True,
    close_fig=False,
    file_name=None,
    description="",
    color_map=COLOR_MAP_FOR_CLIN_SIG,
):
    """
    TODO:
        - check axis limits
        - arg to choose nt or aa variant notation
        - colouring
        - should output file be an argument?
        - should save file be outside of separate function?
        - i_pool can be None and check if multiple pools for gene, and warn merging pools

    """
    if not plot_replicates and not plot_combined:
        raise ValueError(
            "At least one of plot_replicates or plot_combined must be True."
        )
    df = (
        df.loc[(df["symbol"] == gene_name) & (df["pool_id"] == i_pool), :]
        .copy()
        .sort_values(
            ["interactor_symbol", "var_id"],
            key=lambda x: x.apply(sort_var) if x.name == "var_id" else x,
        )
    )
    if df.empty:
        return
    n_replicates = _get_n_replicates(df)
    if include_empty_AD:
        partners = df["interactor_symbol"].unique()
    else:
        partners = [
            p
            for p in df["interactor_symbol"].unique()
            if p not in ["empty-AD", "empty_AD", "emptyAD"]
        ]
    n_partners = len(partners)
    variants = df["var_id"].unique()
    variants = [v for v in variants if v != "WT"]
    fig, axs = plt.subplots(nrows=1, ncols=n_partners, squeeze=False)
    fig.set_size_inches(w=n_partners * 3, h=len(variants) * 0.4)
    for i, (partner_name, ax) in enumerate(zip(partners, axs.flatten())):
        if plot_replicates:
            for i_replicate in range(1, n_replicates + 1):
                _plot_log2fc_with_error_for_interactor(
                    df,
                    partner_name,
                    variants,
                    rep=i_replicate,
                    ax=ax,
                    label=f"Replicate {i_replicate}",
                    color_map=color_map,
                )
        if plot_combined:
            _plot_log2fc_with_error_for_interactor(
                df,
                partner_name,
                variants,
                ax=ax,
                label="Combined estimate",
                color_map=color_map,
            )
        ax.set_ylim(-0.5, len(variants) - 0.5)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.axvline(x=0, color="black", linewidth=1)
        if i > 0:
            ax.set_yticklabels([])
        for loc in ["top", "left", "right"]:
            ax.spines[loc].set_visible(False)
        ax.yaxis.set_tick_params(length=0)
        ax.set_title(partner_name)
        ax.set_xlim(-12, 5)
        ax.set_xlabel("log2FC allele vs WT")
    plt.subplots_adjust(wspace=0.3)
    axs[0, 0].text(
        x=-12,
        y=-0.5,
        s=f"$\\mathbf{{{gene_name}}}$",
        transform=axs[0, 0].transData,
        ha="right",
        va="bottom",
    )
    if plot_replicates:
        axs[0, -1].legend(bbox_to_anchor=(1.05, 0), loc="lower left")
    out_dir = Path(f"../output/figures/{df['experiment'].iloc[0]}/per_gene")
    out_dir.mkdir(parents=True, exist_ok=True)
    if file_name is None:
        file_name = f"{gene_name}_pool_{i_pool}_pooled_Y2H_results{description}.pdf"
    fig.savefig(out_dir / file_name, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_pooled_Y2H_results_interactive(
    df,
    gene_name,
    i_pool=None,
    plot_combined=True,
    plot_replicates=False,
    include_empty_AD=False,
    color_map=COLOR_MAP_FOR_CLIN_SIG,
):
    """
    TODO:
        - get count data for the merge cases
        - option to do separate panels
        - option for separate replicates
    """
    if not plot_replicates and not plot_combined:
        raise ValueError(
            "At least one of plot_replicates or plot_combined must be True."
        )

    if i_pool is None:
        pool_selection = True
        is_multiple_pools_per_gene = False
    else:
        pool_selection = df["pool_id"] == i_pool
        is_multiple_pools_per_gene = (
            df.loc[df["symbol"] == gene_name, "pool_id"].nunique() > 1
        )
    df = (
        df.loc[(df["symbol"] == gene_name) & pool_selection, :]
        .copy()
        .sort_values(
            ["interactor_symbol", "var_id"],
            key=lambda x: x.apply(sort_var) if x.name == "var_id" else x,
        )
    )

    # For plot merged across pools and experiments
    if i_pool is None:
        df_full = df.copy()  # TODO: keep for the read counts - may need to make a column to contain string for hover
        df = df.loc[~df["is_replaced_by_merged_stat"], :]
        nulls_not_merged = (
            (df["var_id"] != "WT")
            & df.duplicated(
                subset=["symbol", "var_id", "interactor_symbol"], keep=False
            )
            & df["log2FC_combined"].isnull()
        )
        df = df.loc[~nulls_not_merged, :]

    if df.empty:
        print(f"WARNING: No data for {gene_name} pool {i_pool}")
        return

    n_replicates = _get_n_replicates(df)

    if not include_empty_AD:
        df = df.loc[df["interactor_id"] != EMPTY_AD_ID, :]
    for interactor in df["interactor_symbol"].unique():
        for i in range(1, n_replicates + 1):
            for media in ["lw", "3at"]:
                df.loc[
                    df["interactor_symbol"] == interactor, f"read_cnt_{media}{i}_wt"
                ] = df.loc[
                    (df["var_id"] == "WT") & (df["interactor_symbol"] == interactor),
                    f"read_cnt_{media}{i}",
                ].iloc[0]
    for i in range(1, n_replicates + 1):
        for media in ["lw", "3at"]:
            df[f"read_cnt_{media}{i}_wt"] = df[f"read_cnt_{media}{i}_wt"].astype(
                pd.Int64Dtype()
            )

    def _hover_content(row):
        return f"""</b>
        <b>{row["symbol"]} {row["var_id"]}</b><br>
        Interactor: <b>{row["interactor_symbol"]}</b><br>
        log2FC: {row["log2FC_combined"]:.2f} ± {row["error_log2FC_combined"]:.2f}
        <br><br>
        Read counts:<br>
        <b>Replicate    -LW  -LWH + 1mM 3AT</b><br>
        1    Var: {row["read_cnt_lw1"]:>6}{row["read_cnt_3at1"]:>10}<br>
              WT: {row["read_cnt_lw1_wt"]:>6}{row["read_cnt_3at1_wt"]:>10}<br>
        <br>
        2    Var: {row["read_cnt_lw2"]:>6}{row["read_cnt_3at2"]:>10}<br>
              WT: {row["read_cnt_lw2_wt"]:>6}{row["read_cnt_3at2_wt"]:>10}<br>
        <br>
        3    Var: {row["read_cnt_lw3"]:>6}{row["read_cnt_3at3"]:>10}<br>
              WT: {row["read_cnt_lw3_wt"]:>6}{row["read_cnt_3at3_wt"]:>10}<br>
        """

    df["hover_data"] = df.apply(_hover_content, axis=1)

    fig = px.scatter(
        data_frame=df.loc[df["var_id"] != "WT", :],
        x="log2FC_combined",
        y="var_id",
        error_x="error_log2FC_combined",
        color="interactor_symbol",
        # see colormap options: https://plotly.com/python/discrete-color/
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_name="hover_data",
        hover_data={
            "log2FC_combined": False,
            "interactor_symbol": False,
            "error_log2FC_combined": False,
            "var_id": False,
        },
    )
    fig.add_vline(x=0, line_width=1, line_color="black")

    pool_text = f" pool {i_pool}" if is_multiple_pools_per_gene else ""
    if df["experiment"].nunique() > 1:
        title = f"{gene_name} pooled Y2H results"
    else:
        title = f"{df['experiment'].iloc[0]}<br>{gene_name}{pool_text}"
    fig.update_layout(
        title=title,
        xaxis_title="log2FC allele vs WT",
        yaxis_title="",
        template="plotly_white",
        legend=dict(title="Interactor"),
        xaxis=dict(range=[-12, 5]),
    )
    if "cv_clinical_significance_clean" in df.columns:
        variant_to_color = (
            df.set_index("var_id")["cv_clinical_significance_clean"]
            .fillna("Other")
            .map(color_map)
            .to_dict()
        )
    else:
        print("WARNING: missing cv_clinical_significance_clean column")
        variant_to_color = {}
    y_labels = df.loc[df["var_id"] != "WT", "var_id"].unique()
    tick_vals = list(range(len(y_labels)))  # positions along y-axis
    tick_text = [
        f'<span style="color:{variant_to_color.get(lbl, "black")}">{lbl}</span>'
        for lbl in y_labels
    ]
    fig.update_yaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        autorange="reversed",
    )
    fig.update_traces(hoverlabel=dict(font_family="monospace"))
    return fig


def plot_pathogenic_vs_benign_log2FC_per_variant_PPI_measurement_hist(
    df, rng=(-11, 11), n_bins=220
):
    df = df.loc[df["interactor_symbol"] != "empty-AD"].copy()
    fig, ax = plt.subplots()
    ax.hist(
        df.loc[
            df["cv_clinical_significance_clean"] == "Pathogenic", "log2FC_combined"
        ].values,
        range=rng,
        bins=n_bins,
        label=f"Pathogenic (N={(df['cv_clinical_significance_clean'] == 'Pathogenic').sum()})",
        color=COLOR_PATHOGENIC,
        alpha=0.7,
    )
    ax.hist(
        df.loc[
            df["cv_clinical_significance_clean"] == "Benign", "log2FC_combined"
        ].values,
        range=rng,
        bins=n_bins,
        label=f"Benign (N={(df['cv_clinical_significance_clean'] == 'Benign').sum()})",
        color=COLOR_BENIGN,
        alpha=0.7,
    )
    ax.legend()
    ax.set_title("Across all PPIs")
    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("Number of variant-PPI measurements")


def plot_pathogenic_vs_benign_log2FC_per_variant_hist(df):
    df = df.loc[df["interactor_symbol"] != "empty-AD"].copy()
    fig, ax = plt.subplots()
    rng = (-11, 11)
    n_bins = 22
    for clin_sig, color in [("Pathogenic", COLOR_PATHOGENIC), ("Benign", COLOR_BENIGN)]:
        log2fc = (
            df.loc[df["cv_clinical_significance_clean"] == clin_sig, :]
            .groupby(["symbol", "var_id"])["log2FC_combined"]
            .mean()
            .values
        )

        ax.hist(
            log2fc,
            range=rng,
            bins=n_bins,
            label=f"{clin_sig} (N = {len(log2fc)})",
            color=color,
            alpha=0.7,
        )
    ax.legend()
    ax.set_title("Average per variant")
    ax.set_xlabel("Mean Log2 Fold Change across PPIs")
    ax.set_ylabel("Number of variants")


def plot_ROC_curve(df, vars_to_plot):
    """
    TODO:
        - add AUC number

    """
    df = df.loc[
        df["cv_clinical_significance_clean"].isin(["Pathogenic", "Benign"])
        & df[vars_to_plot].notnull().all(axis=1)
    ].copy()

    df["is_pathogenic"] = df["cv_clinical_significance_clean"] == "Pathogenic"
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    for var in vars_to_plot:
        x, y = [], []
        target = df["is_pathogenic"]
        for cutoff in sorted(df[var].values):
            p = df[var] <= cutoff
            tp = target[p].sum()
            fp = (~target[p]).sum()
            tn = (~target[~p]).sum()
            fn = target[~p].sum()
            y.append(tp / (tp + fn))
            x.append(1 - (tn / (tn + fp)))
        ax.plot(x, y, label=var)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")
    ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
    fig.savefig(
        f"../output/figures/{df['experiment'].iloc[0]}/ROC.pdf", bbox_inches="tight"
    )


def plot_balanced_PR_curve(df, vars_to_plot):
    """
    Based on Fritz et al.'s paper with balanced and monotonized PR curves

    TODO:
        - add balanced PR number

    """
    df = df.loc[
        df["cv_clinical_significance_clean"].isin(["Pathogenic", "Benign"])
        & df[vars_to_plot].notnull().all(axis=1)
    ].copy()

    df["is_pathogenic"] = df["cv_clinical_significance_clean"] == "Pathogenic"
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)
    prior = df["is_pathogenic"].sum() / df.shape[0]
    for var in vars_to_plot:
        x, y = [], []
        target = df["is_pathogenic"]
        for cutoff in sorted(df[var].values, reverse=True):
            p = df[var] <= cutoff
            tp = target[p].sum()
            fp = (~target[p]).sum()
            tn = (~target[~p]).sum()
            fn = target[~p].sum()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            balanced_precision = (precision * (1.0 - prior)) / (
                precision * (1 - prior) + (1 - precision) * prior
            )
            x.append(recall)
            if len(y) == 0 or balanced_precision >= y[-1]:
                y.append(balanced_precision)
            else:
                y.append(y[-1])
        x.append(0)
        y.append(1)
        ax.plot(x, y, label=var)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
    ax.set_xticks(
        np.linspace(0, 1, 6),
    )
    ax.set_xticks(np.linspace(0, 1, 11), minor=True)
    ax.set_xticklabels([f"{x:.0%}" for x in ax.get_xticks()])
    ax.set_yticks(
        np.linspace(0, 1, 6),
    )
    ax.set_yticks(np.linspace(0, 1, 11), minor=True)
    ax.set_yticklabels([f"{y:.0%}" for y in ax.get_yticks()])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(
        f"../output/figures/{df['experiment'].iloc[0]}/balanced_PR.pdf",
        bbox_inches="tight",
    )