"""
TODO

- remove need for loop over stat type in single replicate function
- combine stats into single function
- log filtering output
- input table validation function

"""

import re
import functools
import json
from pathlib import Path

import numpy as np
from scipy import stats, optimize
import pandas as pd
from statsmodels.stats.meta_analysis import combine_effects
from Bio.Align import substitution_matrices
from Bio.Data import IUPACData
import pyBigWig

from mapping_to_reference_sequences import (
    add_uniprot_aa_pos_mapping,
    test_uniprot_mappings,
)
from utils import _get_n_replicates
from config import (
    MIN_READS_IN_LW,
    MIN_READS_WT_3AT,
    MIN_POOL_FRACTION_RELATIVE_TO_EXPECTED,
)
from diseases import add_clinvar_phenotypes, add_mondo_ids, add_mode_of_inheritance
from plotting import plot_filtering_summary, plot_growth_scores_by_gene_pies
from gnomad_api import add_allele_frequency
from alphafold_monomer_properties import add_alphafold_monomer_properties
from variant_effect_predictors import add_alphamissense_column
from alphafold_dimer_models import add_AF3_dimer_info


EMPTY_AD_ID = 0


def load_combined_pooled_Y2H_dataset(
    drop_superceded=True, add_alphafold=True, add_perturbation_llr=True
):
    """
    Dataset as of 2026-01-27. Doesn't use Hongtao's data, or the CCM2 data.

    TODO:
        - implement splicing filter
        - fix cases of multiple nt changes mapping to same aa change
        - remove genes with poor mapping to uniprot
        - remove KRTAPs etc???
        - change to exclude list of experiments, since shorter

    """
    experiments = [
        "LITAF",
        "smad3_gel_purify",
        # "smad3-liquid-pooled-y2h",
        "TRIM32",
        "pilot3",
        "HsVcPPIP01a1",
        "HsVcPPIP01a2",
        "HsVcPPIP01a6",
        "HsVcPPIP01v1AD",
        "HsVcPPIP01v6AD",
        "HsVcPPIP02a1",
        "HsVcPPIP02rdo",
        "HsVcPPIP02a2rd",
        "HsVcPPIP02a6rd",
        "HsVcPPIP02v1AD",
        "HsVcPillar2a1",
        "HsVcPillar2a2",
        "HsVcPillar2a6",
        "HsVcPPILacVa1",
        "HsVcPPILacVa1AD",
        "HsVcPPILacVa2",  # NOTE I think we considered not including this one because the reproducibility was bad
        "HsVcPPILacVa6",
        #'HsVcPPILiqPt1',
        "HsVcPPIMLHSTAT",
    ]
    liquid_media_expeiriments = ["HsVcPPILiqPt1", "smad3-liquid-pooled-y2h"]

    df = pd.concat(
        [
            load_pooled_y2h_experiment(
                e,
                is_liquid_media=(e in liquid_media_expeiriments),
                add_perturbation_llr=add_perturbation_llr,
            )
            for e in experiments
        ]
    )

    # TODO: this is due to the liquid media experiment. Fix in upstream functions.
    df = df.loc[df["interactor_id"] != EMPTY_AD_ID, :]

    df = combine_results_across_experiments(df, drop_superceded=drop_superceded)
    if add_alphafold:
        df = add_alphamissense_column(df)
        df = add_alphafold_monomer_properties(df)
        df = add_AF3_dimer_info(df)
    return df


def load_pooled_y2h_experiment(
    experiment_id, is_liquid_media=False, add_perturbation_llr=True
):
    input_data_dir = Path("../data/internal")
    path_3at = input_data_dir / f"pooled-Y2H_read-counts_{experiment_id}_3AT.tsv"
    path_lw = input_data_dir / f"pooled-Y2H_read-counts_{experiment_id}_LW.tsv"
    if is_liquid_media:
        path_growth_scores = None
    else:
        path_growth_scores = input_data_dir / f"{experiment_id}_plate_map_exp_info.tsv"

    return read_pooled_Y2H_read_counts(
        file_path_3at=path_3at,
        file_path_lw=path_lw,
        file_path_growth_scores=path_growth_scores,
        add_perturbation_llr=add_perturbation_llr,
    )


def read_pooled_Y2H_read_counts(
    file_path_lw,
    file_path_3at,
    file_path_growth_scores=None,
    doubled_wt=True,
    override_missing_variants_error=False,
    add_variant_info=True,
    filter_data=True,
    add_splice_predictions=False,
    add_perturbation_llr=True,
    add_cross_references=True,
    make_plots=False,
    variants_data_path="../data/internal/CCSB_variants_info.tsv",
    verbose=False,
):
    """
    TODO:

    - Implement input table validation

    """
    print(f"Reading counts from {file_path_lw} and {file_path_3at}")
    df = pd.concat(
        [pd.read_csv(file_path_lw, sep="\t"), pd.read_csv(file_path_3at, sep="\t")]
    )
    pivot_cols = ["read_count"]
    id_cols = [col for col in df.columns if col not in pivot_cols + ["media"]]
    df = df.pivot(index=id_cols, columns="media", values=pivot_cols).reset_index()
    df.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in df.columns
    ]
    df = df.rename(
        columns={"read_count_LW": "read_cnt_lw", "read_count_3AT": "read_cnt_3at"}
    )
    df["orf_id_wt"] = df["ccsb_orf_id_reference"].apply(
        lambda x: int(x[len("CCSBORF") :])
    )
    df["interactor_id"] = df["ccsb_orf_id_interactor"].apply(
        lambda x: int(x[len("CCSBORF") :])
    )
    df.loc[df["nt_change"] == "REF", "nt_change"] = pd.NA
    if "ensembl_gene_id" in df.columns:
        df = df.drop(columns=["ensembl_gene_id", "spdi"])

    validate_input_counts_table(df, file_path=(file_path_3at, file_path_lw))

    # NOTE: currently assuming a fixed number of replicates across all genes etc.
    n_replicates = df["repeat_id"].nunique()
    # TODO: check if all genes have the same number of replicates
    df["var_id"] = df["nt_change"].fillna("WT")
    pivot_cols = ["read_cnt_lw", "read_cnt_3at"]
    id_cols = [col for col in df.columns if col not in pivot_cols + ["repeat_id"]]
    df = df.pivot(index=id_cols, columns="repeat_id", values=pivot_cols).reset_index()
    # TODO: switch to underscore
    df.columns = [f"{col[0]}{col[1]}" for col in df.columns]

    # Add WT read counts as columns to variant rows
    # NOTE: assummes a single WT per pool
    merge_columns = ["experiment", "symbol", "pool_id", "interactor_id"]
    count_columns = [f"read_cnt_lw{i}" for i in range(1, n_replicates + 1)] + [
        f"read_cnt_3at{i}" for i in range(1, n_replicates + 1)
    ]
    wt = (
        df.loc[df["var_id"] == "WT", merge_columns + count_columns]
        .copy()
        .rename(
            columns={c: c.replace("read_cnt", "wt_read_cnt") for c in count_columns}
        )
    )
    n_rows_b4 = df.shape[0]
    df = pd.merge(df, wt, on=merge_columns, how="left")
    if df.shape[0] != n_rows_b4:
        raise UserWarning("Merging WT read counts changed number of rows")

    for i in range(1, n_replicates + 1):
        df[f"total_read_lw{i}"] = df.groupby(["symbol", "pool_id", "interactor_id"])[
            f"read_cnt_lw{i}"
        ].transform("sum")
        df[f"total_read_3at{i}"] = df.groupby(["symbol", "pool_id", "interactor_id"])[
            f"read_cnt_3at{i}"
        ].transform("sum")
        df["pct_read_cnt_3at" + str(i)] = (
            df["read_cnt_3at" + str(i)] / df["total_read_3at" + str(i)] * 100
        )
        df["pct_read_cnt_lw" + str(i)] = (
            df["read_cnt_lw" + str(i)] / df["total_read_lw" + str(i)] * 100
        )

    # adding half count is the Haldane-Anscombe correction for dealing with 0 counts
    for i_rep in range(1, n_replicates + 1):
        a = df[f"wt_read_cnt_lw{i_rep}"] + 0.5
        b = df[f"wt_read_cnt_3at{i_rep}"] + 0.5
        c = df[f"read_cnt_lw{i_rep}"] + 0.5
        d = df[f"read_cnt_3at{i_rep}"] + 0.5
        odds_ratio = (a / b) / (c / d)
        logOR = np.log(odds_ratio)
        error_logOR = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        df[f"log2FC_{i_rep}"] = np.log2(odds_ratio)
        scaling_factor = (df[f"log2FC_{i_rep}"] / logOR).where(logOR != 0, 1)
        df[f"error_log2FC_{i_rep}"] = error_logOR * scaling_factor

        n_var = df.groupby(["symbol", "pool_id"])["var_id"].transform("nunique")
        exp_pct = 100 / (n_var + 1) if doubled_wt == True else 100 / n_var
        fail_filter = (
            (df[f"read_cnt_lw{i_rep}"] < MIN_READS_IN_LW)
            | (df[f"wt_read_cnt_lw{i_rep}"] < MIN_READS_IN_LW)
            | (df[f"wt_read_cnt_3at{i_rep}"] < MIN_READS_WT_3AT)
            | (
                df[f"pct_read_cnt_lw{i_rep}"]
                <= (exp_pct * MIN_POOL_FRACTION_RELATIVE_TO_EXPECTED)
            )
        )
        df.loc[fail_filter | (df["var_id"] == "WT"), f"log2FC_{i_rep}"] = np.nan
        df.loc[fail_filter | (df["var_id"] == "WT"), f"error_log2FC_{i_rep}"] = np.nan

    df = pd.concat(
        [
            df,
            df.apply(calc_stats_combined_across_replicates, n_rep=n_replicates, axis=1),
        ],
        axis=1,
    )
    df = apply_floored_error(df)
    df["z_score"] = df["log2FC_combined"] / df["error_log2FC_combined"]
    if add_perturbation_llr:
        df = _add_perturbation_LLR(df)
        df["perturbation_status"] = df["perturbation_LLR"].apply(perturbation_status)
    if file_path_growth_scores is not None:
        df = _add_growth_scores(df, file_path_growth_scores, make_plots=make_plots)
    if add_variant_info:
        df = _add_variant_info(
            df,
            variants_data_path=variants_data_path,
            override_missing_variants_error=override_missing_variants_error,
        )

    # print a warning if filter_data  add_splice_predictions are both True
    if filter_data and add_splice_predictions:
        print("WARNING: filter_data and add_splice_predictions are both set to True")

    if filter_data:
        df = _filter_data(df, make_plots=make_plots, verbose=verbose)

    if add_splice_predictions:
        df = _add_splice_predictions(df)

    if add_cross_references:
        df = _add_cross_reference_ids_for_interactors(df)
    return df


def _add_splice_predictions(df):
    """
    add SpliceAI predictions to df, print warnings if necessary
    """

    # load spliceai 500nt distance dataset
    sdf = pd.read_csv(
        "../data/internal/ccsb_mutations_with_spliceai_scores_dist_500.tsv",
        delimiter="\t",
    )

    # filter out uncloned splicing predictions
    sdf = sdf[sdf.cloned == 1].copy()

    # add a unique identifier column
    df["key_col"] = df.orf_id_wt.astype(str) + "@" + df.nt_change
    sdf["key_col"] = sdf.orf_id.astype(str) + "@" + sdf.nt_change
    sdf.drop_duplicates(subset=["key_col"], inplace=True)

    # define variant sets
    df_variants = set([i for i in set(df.key_col.unique()) if not pd.isna(i)])

    # define variant set that's been tested for splicing effects
    tested_variants = set(sdf.key_col.unique().tolist())

    cols_to_add = [
        "DS_AG",
        "DS_AL",
        "DS_DG",
        "DS_DL",
        "DP_AG",
        "DP_AL",
        "DP_DG",
        "DP_DL",
        "any_DS_above_0.2",
        "any_DS_above_0.5",
        "any_DS_above_0.8",
    ]
    cd = {att: [] for att in cols_to_add}
    for _, row in df.iterrows():
        if pd.isna(row["key_col"]):
            for att in cols_to_add:
                cd[att].append("NULL")
            continue

        sai_row = sdf[sdf["key_col"] == row["key_col"]]
        # add empty row to cd if a df variant isn't found in SpliceAI data
        if sai_row.shape[0] == 0:
            for att in cols_to_add:
                cd[att].append("NULL")
            continue

        for att in cols_to_add:
            val = sai_row[att].tolist()[0]
            cd[att].append(val)

    for att, col_data in cd.items():
        df[att] = col_data

    # print a warning if variants appear in the df that weren't tested with SpliceAI
    n_not_tested = len(df_variants.difference(tested_variants))
    if n_not_tested > 0:
        print(
            "WARNING: "
            + str(n_not_tested)
            + " variants in the input df weren't tested with SpliceAI"
        )

    # print a warning if DS_AG column is empty for any df rows and variants
    null_ds_ag = df[
        (pd.isna(df.DS_AG)) & (df["pos"] != 0)
    ]  # WT rows have NULL for DS_AG so are excluded at this point (i.e. NaN != NULL)
    if null_ds_ag.shape[0] > 0:
        n_null_vars = null_ds_ag["key_col"].nunique()
        print(
            "WARNING (_add_splice_predictions): "
            + str(null_ds_ag.shape[0])
            + " rows ("
            + str(n_null_vars)
            + " variants) were not successfully tested with SpliceAI, so their effect on splicing is unknown"
        )

    df = df[[col for col in df.columns if col != "key_col"]].copy()

    df[cols_to_add] = df[cols_to_add].fillna("NULL")

    return df


def validate_input_counts_table(df, file_path=None):
    if df.shape[0] == 0:
        raise ValueError(f"Input read counts table: {file_path} is empty")
    # check empty-AD ID and symbol consistency
    # integer counts
    # non-negative counts
    # required columns present
    # no unexpected missing values
    # each PPI/variant has a row for each replicate
    # expect at least three replicates


def validate_growth_scores_table(df, file_path=None):
    if df.shape[0] == 0:
        raise ValueError(f"Growth scores table: {file_path} is empty")
    if "y2h_score" not in df.columns:
        raise ValueError(f"Growth scores table: {file_path} missing 'y2h_score' column")
    if df["y2h_score"].isnull().all():
        raise ValueError(
            f"Growth scores table: {file_path} missing all values in 'y2h_score' column"
        )
    # Growth scores between 0-4 + NA


@functools.lru_cache()
def load_variant_info(variants_data_path="../data/internal/CCSB_variants_info.tsv"):
    vars = pd.read_csv(variants_data_path, sep="\t", low_memory=False)

    # since it's there in the ID mapping table
    vars = vars.drop(columns=["ensembl_gene_id", "ensembl_protein_id"], errors="ignore")

    if "clinical_significance_simple" in vars.columns:
        # TODO: fix this in the input table
        vars["clinical_significance_simple"] = vars[
            "clinical_significance_simple"
        ].apply(
            lambda x: {
                "pathogenic": "Pathogenic",
                "benign": "Benign",
                "other": "Other",
            }.get(x, x)
        )
    vars["orf_id_wt"] = vars["orf_id"]

    blosum62 = substitution_matrices.load("BLOSUM62")
    three_to_one = IUPACData.protein_letters_3to1
    # since inconsistent capitalization in input table
    vars["ref_aa"] = vars["ref_aa"].str.title()
    vars["alt_aa"] = vars["alt_aa"].str.title()
    vars["ref_aa_1letter"] = vars["ref_aa"].map(three_to_one)
    vars["alt_aa_1letter"] = vars["alt_aa"].map(three_to_one)

    vars = add_uniprot_aa_pos_mapping(vars)
    test_uniprot_mappings(vars)

    vars["blosum62"] = vars.apply(
        lambda row: (
            blosum62[row["ref_aa_1letter"], row["alt_aa_1letter"]]
            if pd.notnull(row["alt_aa_1letter"])
            else np.nan
        ),
        axis=1,
    )
    if vars["blosum62"].isnull().any():
        print(
            "WARNING: Problem with BLOSUM62 lookup; check all amino acid codes are valid."
        )

    phylop = pyBigWig.open("../data/external/hg38.phyloP100way.bw")

    def get_phyloP(row):
        return phylop.values(
            "chr" + row["chr"], row["chr_pos_38"] - 1, row["chr_pos_38"]
        )[0]

    if "chr" in vars.columns and "chr_pos_38" in vars.columns:
        vars["phyloP"] = vars.apply(
            get_phyloP,
            axis=1,
        )

    vars = add_allele_frequency(vars)

    if "cv_allele_id" in vars.columns:
        vars = add_clinvar_phenotypes(vars)

    if "cv_phenotype_ids" in vars.columns:
        vars = add_mondo_ids(vars)

    if "mondo_ids" in vars.columns:
        vars = add_mode_of_inheritance(vars)

    return vars


def _add_variant_info(
    df,
    variants_data_path="../data/internal/CCSB_variants_info.tsv",
    override_missing_variants_error=False,
):
    """
    TODO:
        - convert aa_change to uniprot_seq using SPDI

    """
    df = df.copy()
    vars = load_variant_info(variants_data_path)

    merge_columns = ["nt_change", "orf_id_wt"]
    columns_to_add = [
        "aa_change",
        "ref_aa",
        "alt_aa",
        "aa_change_hgvs",
        "aa_change_cloned_orf",
        "aa_change_uniprot_canonical",
        "aa_change_uniprot_isoform",
        "mutation_id",
        "spdi",
        "ensembl_protein_id",
        "ensembl_gene_id",
        "uniprot_ac",
        "uniprot_isoform_ac",
        "wt_orf_matches_uniprot_canonical",
        "n_mismatches_with_uniprot_sequence",
        "pct_identity_with_uniprot_sequence",
        "wt_orf_uniprot_sequence_match_details",
        "hgnc_symbol",
        "hgnc_id",
        "hgmd_id",
        "hgmd_clinical_significance",
        "hgmd_phenotype",
        "cv_allele_id",
        "cv_clinical_significance",
        "cv_clinical_significance_clean",
        "clinical_significance_simple",
        "cv_phenotypes",
        "cv_review_status",
        "cv_origin",
        "cv_origin_simple",
        "mondo_ids",
        "n_mondo_ids",
        "filtered_mondo_ids",
        "filtered_mondo_phenotypes",
        "n_filtered_mondo_ids",
        "high_level_mondo_terms",
        "blosum62",
        "phyloP",
        "allele_frequency",
        "moi",
    ]
    columns_to_add = [c for c in columns_to_add if c in vars.columns]
    n_rows_b4 = df.shape[0]
    df = pd.merge(
        df, vars.loc[:, merge_columns + columns_to_add], on=merge_columns, how="left"
    )
    if df.shape[0] != n_rows_b4:
        msg = f"""Warning: {df.shape[0] - n_rows_b4} new rows added during merge.
                Probably, this could come from either:
                1. Mistake duplicates in table
                2. Cases where one ORF maps to multiple genes, hence variants, e.g. HBA1/2
                3. Multiple clones of the same variant existing
                """
        raise UserWarning(msg)
    if df.loc[df["var_id"] != "WT", "aa_change"].isnull().any():
        msg = "Unexpected missing values found for variants\n"
        msg += "You might need to update variants table from the database and restart python to clear cache.\n"
        msg += f"{df.loc[(df['var_id'] != 'WT') & df['aa_change'].isnull(), ['symbol', 'var_id']].drop_duplicates().to_string(index=False)}"
        if not override_missing_variants_error:
            raise UserWarning(msg)
        else:
            print(msg)
            df = df.loc[(df["var_id"] == "WT") | df["aa_change"].notnull(), :]
    df["aa_pos"] = df["aa_change"].str.slice(3, -3).astype("Int64")
    df["aa_pos_cloned_orf"] = (
        df["aa_change_cloned_orf"].str.slice(3, -3).astype("Int64")
    )
    df["aa_pos_uniprot_canonical"] = (
        df["aa_change_uniprot_canonical"].str.slice(3, -3).astype("Int64")
    )
    df["aa_pos_uniprot_isoform"] = (
        df["aa_change_uniprot_isoform"].str.slice(3, -3).astype("Int64")
    )
    df["var_id"] = df["aa_change"].fillna("WT")

    # This is not very tidy but we need to fill in missing IDs for WT rows
    for id_col, dtype in [
        ("uniprot_ac", "string"),
        ("uniprot_isoform_ac", "string"),
        ("ensembl_gene_id", "string"),
        ("ensembl_protein_id", "string"),
        ("hgnc_symbol", "string"),
        ("hgnc_id", "string"),
        ("wt_orf_matches_uniprot_canonical", "boolean"),
        ("n_mismatches_with_uniprot_sequence", "Int64"),
        ("pct_identity_with_uniprot_sequence", "Float64"),
        ("wt_orf_uniprot_sequence_match_details", "string"),
    ]:
        df[id_col] = df[id_col].astype(dtype)  # To supress a warning
        df[id_col] = df.groupby("orf_id_wt")[id_col].transform(
            lambda x: x.bfill().ffill()
        )

    if df["uniprot_ac"].isnull().any():
        print(
            f"WARNING: Missing values in UniProt AC mapping {df.loc[df['uniprot_ac'].isnull(), 'symbol'].unique()}"
        )

    return df


def _add_growth_scores(df, file_path_growth_scores, make_plots=False):
    """
    TODO: include well-mapped read count data

    """
    df = df.copy()
    tb = pd.read_csv(file_path_growth_scores, sep="\t")
    validate_growth_scores_table(tb, file_path=file_path_growth_scores)
    merge_columns = ["symbol", "pool_id", "interactor_id"]
    df_keys = set(map(tuple, df[merge_columns].drop_duplicates().to_numpy()))
    tb_keys = set(map(tuple, tb[merge_columns].drop_duplicates().to_numpy()))
    if df_keys != tb_keys:
        msg = f"Mismatch between counts and growth score tables {file_path_growth_scores}\n"
        diff_a = df_keys.difference(tb_keys)
        diff_b = tb_keys.difference(df_keys)
        if diff_a:
            msg += f"Entries in counts but not growth scores:\n{pd.DataFrame(list(diff_a), columns=merge_columns).to_string(index=False)}\n"
        if diff_b:
            msg += f"Entries in growth scores but not counts:\n{pd.DataFrame(list(diff_b), columns=merge_columns).to_string(index=False)}\n"
        raise ValueError(msg)

    # TODO validate input table
    if make_plots:
        plot_growth_scores_by_gene_pies(
            tb,
            output_dir=f"../output/figures/{df['experiment'].iloc[0]}",
            close_fig=True,
        )
    tb["well_minus_mapped"] = tb["well_read_cnt"] - tb["mapped_read_cnt"]

    for rep in range(1, 4):
        for media in ["LW", "3AT"]:
            df = pd.merge(
                df,
                tb.loc[
                    (tb["repeat_id"] == rep) & (tb["media"] == media),
                    merge_columns + ["y2h_score"],
                ].rename(columns={"y2h_score": f"growth_score_{media}_{rep}"}),
                on=merge_columns,
                how="left",
            )

    tb["pct_well_to_mapped"] = (tb["mapped_read_cnt"] / tb["well_read_cnt"]) * 100

    merge_columns = ["symbol", "pool_id", "interactor_id"]
    for rep in range(1, 4):
        for media in ["LW", "3AT"]:
            df = pd.merge(
                df,
                tb.loc[
                    (tb["media"] == media) & (tb["repeat_id"] == rep),
                    merge_columns + ["pct_well_to_mapped"],
                ].rename(
                    columns={
                        "pct_well_to_mapped": f"pct_well_to_mapped_{media}_rep{rep}"
                    }
                ),
                how="left",
                on=merge_columns,
            )

    n_rep = _get_n_replicates(df)
    df["growth_score_3AT_min"] = df[
        [f"growth_score_3AT_{i}" for i in range(1, n_rep + 1)]
    ].min(axis=1)
    df["growth_score_3AT_max"] = df[
        [f"growth_score_3AT_{i}" for i in range(1, n_rep + 1)]
    ].max(axis=1)
    df["growth_score_3AT_median"] = df[
        [f"growth_score_3AT_{i}" for i in range(1, n_rep + 1)]
    ].median(axis=1)

    return df


def _filter_data(
    df,
    MIN_READS_IN_LW=MIN_READS_IN_LW,
    MIN_READS_WT_3AT=MIN_READS_WT_3AT,
    verbose=False,
    make_plots=False,
):
    """
    TODO:
        - add flag for splicing variants

    """
    if df.shape[0] == 0:
        raise ValueError("Empty dataframe")
    df = df.copy()
    n_rep = _get_n_replicates(df)
    experiment_id = df["experiment"].iloc[0]
    log = {}

    def count_at_different_levels(df):
        df = df.copy()
        df = df.loc[(df["interactor_id"] != EMPTY_AD_ID) & (df["var_id"] != "WT"), :]
        n_reps = _get_n_replicates(df)
        # n_var_ppi_reps = df.shape[0] * n_reps
        n_var_ppi_combined = df.shape[0]
        n_ppis_per_pool = (
            df.loc[:, ["symbol", "pool_id", "interactor_symbol"]]
            .drop_duplicates()
            .shape[0]
        )
        n_ppis = df.loc[:, ["symbol", "interactor_symbol"]].drop_duplicates().shape[0]
        n_genes = df["symbol"].nunique()
        return {
            # need to fix how this is done to get per-replicate counts
            # "Variant PPI replicate measurements": n_var_ppi_reps,
            "Variant PPI combined measurements": n_var_ppi_combined,
            "gene-pool gene PPIs": n_ppis_per_pool,
            "gene gene PPIs": n_ppis,
            "Disease genes": n_genes,
        }

    # Cases where multiple nucleotide variants map to the same a.a. change
    # e.g. GJB2 250G>C and 250G>T both map to Val84Leu
    # TODO: could do a better job here by taking the best measurement
    n_b4 = df.shape[0]

    log["Pre-filtering"] = count_at_different_levels(df)

    df = df.drop_duplicates(
        ["experiment", "symbol", "pool_id", "interactor_symbol", "var_id"]
    )
    if verbose and n_b4 - df.shape[0] > 0:
        print(
            f"Removed {n_b4 - df.shape[0]} rows where multiple tested nucleotide variants map to the same a.a. change"
        )
        log["Single nt variant per a.a. change"] = count_at_different_levels(df)

    # we can't test Met1 variants because of the N-terminal fusions we use
    first_aa_var = df["aa_change"].str.startswith("Met1") & (
        df["aa_change"].str.len() == 7
    )
    if verbose and first_aa_var.any():
        print(f"Removing {first_aa_var.sum()} Met1Xxx variants")
        df = df.loc[~first_aa_var, :]
        log["Removed translation start codon variants"] = count_at_different_levels(df)

    if "growth_score_3AT_median" in df.columns:
        df = df.loc[df["growth_score_3AT_median"] > 0, :]
        log["Median growth score > 0"] = count_at_different_levels(df)
        autoactivators = df.loc[
            (df["interactor_id"] == EMPTY_AD_ID) & (df["growth_score_3AT_median"] >= 3),
            ["experiment", "symbol", "pool_id"],
        ].drop_duplicates()
        df = pd.merge(
            df,
            autoactivators,
            how="left",
            on=["experiment", "symbol", "pool_id"],
            indicator="is_autoactivator",
        ).assign(is_autoactivator=lambda x: x["is_autoactivator"] == "both")
        if verbose:
            print(f"{autoactivators.shape[0]} autoactivating DB pools")
        df = df.loc[~df["is_autoactivator"] & (df["interactor_id"] != EMPTY_AD_ID), :]
        log["Removing autoactivators (median empty-AD growth score ≥ 3)"] = (
            count_at_different_levels(df)
        )

        """
        removing this for now, since we use NA for spotting problems and contamination
        if df["growth_score_3AT_median"].isnull().any():
            msg = f"Unexpected missing growth scores for {df['growth_score_3AT_median'].isnull().mean():.0%} of pairs\n"
            msg += str(
                df.loc[
                    df["growth_score_3AT_median"].isnull(),
                    ["experiment", "symbol", "pool_id", "interactor_symbol"],
                ].drop_duplicates()
            )
            raise UserWarning(msg)
        """
        df = df.loc[df["growth_score_3AT_median"] >= 3, :]
        log["Requiring median growth score ≥ 3)"] = count_at_different_levels(df)

    gte_100_total_3at = (
        df[[f"total_read_3at{i}" for i in range(1, n_rep + 1)]] >= 100
    ).sum(axis=1) >= 2
    gte_min_wt_lw = (
        df[[f"wt_read_cnt_lw{i}" for i in range(1, n_rep + 1)]] >= MIN_READS_IN_LW
    ).sum(axis=1) >= 2
    gte_min_wt_3at = (
        df[[f"wt_read_cnt_3at{i}" for i in range(1, n_rep + 1)]] >= MIN_READS_WT_3AT
    ).sum(axis=1) >= 2
    gte_min_var_lw = (
        df[[f"read_cnt_lw{i}" for i in range(1, n_rep + 1)]] >= MIN_READS_IN_LW
    ).sum(axis=1) >= 2
    log["At least 100 total mapped 3AT reads"] = count_at_different_levels(
        df.loc[gte_100_total_3at]
    )
    log[f"WT has at least {MIN_READS_IN_LW} LW reads"] = count_at_different_levels(
        df.loc[gte_100_total_3at & gte_min_wt_lw]
    )
    log[f"WT has at least {MIN_READS_WT_3AT} 3AT reads"] = count_at_different_levels(
        df.loc[gte_100_total_3at & gte_min_wt_lw & gte_min_wt_3at]
    )
    log[f"Variant has at least {MIN_READS_IN_LW} LW reads"] = count_at_different_levels(
        df.loc[gte_100_total_3at & gte_min_wt_lw & gte_min_wt_3at & gte_min_var_lw]
    )

    # filter for at least one valid measurement per PPI
    n_b4 = df.shape[0]
    df = df.groupby(["experiment", "symbol", "pool_id", "interactor_symbol"]).filter(
        lambda x: (
            x["log2FC_combined"].notnull() | (x["interactor_id"] == EMPTY_AD_ID)
        ).any()
    )
    if verbose and n_b4 > 0:
        print(
            f"Dropped {n_b4 - df.shape[0]} ({(n_b4 - df.shape[0]) / n_b4:.1%}) rows without any measurements per PPI"
        )
    n_b4 = df.shape[0]
    # filter for at least one non empty-AD interactor valid measurement per experiment/gene/pool
    df = df.groupby(["experiment", "symbol", "pool_id"]).filter(
        lambda x: (
            x["log2FC_combined"].notnull() & ~(x["interactor_id"] == EMPTY_AD_ID)
        ).any()
    )
    if verbose and n_b4 > 0:
        print(
            f"Dropped {n_b4 - df.shape[0]} ({(n_b4 - df.shape[0]) / n_b4:.1%}) rows without any measurements per gene"
        )
    n_b4 = df.shape[0]
    df = df.groupby(["experiment", "symbol", "pool_id", "var_id"]).filter(
        lambda x: (
            (x["var_id"] == "WT")
            | (x["log2FC_combined"].notnull() & ~(x["interactor_id"] == EMPTY_AD_ID))
        ).any()
    )
    if verbose and n_b4 > 0:
        print(
            f"Dropped {n_b4 - df.shape[0]} ({(n_b4 - df.shape[0]) / n_b4:.1%}) rows without any valid measurements per variant"
        )

    log["Variant ≥ 10% of expected pool fraction"] = count_at_different_levels(
        df.loc[df["log2FC_combined"].notnull(), :]
    )

    ############splicing

    n_b4 = df.shape[0]
    # load spliceai 500nt distance dataset
    sdf = pd.read_csv(
        "../data/internal/ccsb_mutations_with_spliceai_scores_dist_500.tsv",
        delimiter="\t",
    )

    # filter out uncloned splicing predictions
    sdf = sdf[sdf.cloned == 1].copy()

    # add a unique identifier column
    df["key_col"] = df.orf_id_wt.astype(str) + "@" + df.nt_change
    sdf["key_col"] = sdf.orf_id.astype(str) + "@" + sdf.nt_change
    sdf.drop_duplicates(subset=["key_col"], inplace=True)

    # print a warning if a y2h variant has null SpliceAI predictions, as measrured by an empty DS_AG column
    tdf = sdf[sdf.key_col.isin(df.key_col)]
    null_ds_ag = tdf[pd.isna(tdf.DS_AG)]
    if null_ds_ag.shape[0] > 0:
        n_null_vars = null_ds_ag["key_col"].nunique()
        print(
            "WARNING (_filter_data): "
            + str(n_null_vars)
            + " variants were not successfully tested with SpliceAI, so their effect on splicing is unknown. These variants will NOT be filtered out."
        )

    # define set of variants that we've tested for splicing effects
    splice_tested_set = set([i for i in set(sdf.key_col) if not pd.isna(i)])

    # define set of variants in the df
    df_var_set = set([i for i in set(df.key_col) if not pd.isna(i)])

    # print out warnings if df contains variants not tested for splicing changes
    for variant in df_var_set.difference(splice_tested_set):
        sym, i_var = variant.split("@")
        print(
            "WARNING (_filter_data): variant ("
            + i_var
            + ") for gene ("
            + sym
            + ") not tested for possible splicing effects."
        )

    # filter according to the threshold we decided
    sdf = sdf[sdf["any_DS_above_0.2"] == True].copy()

    # define variants that might affect splicing
    splicing_positive = set(sdf.key_col.unique())

    df_mod = df.copy(deep=True)
    df_mod["splicing_hit"] = df_mod.key_col.isin(splicing_positive)

    # remake df without extraneous cols
    df_mod_filt = df_mod[df_mod.splicing_hit == False][
        [i for i in df_mod.columns if "splicing" not in i if "key_col" not in i]
    ]
    df = df_mod_filt.copy(deep=True)

    if verbose and n_b4 > 0:
        print(
            f"Dropped {n_b4 - df.shape[0]} ({(n_b4 - df.shape[0]) / n_b4:.1%}) rows with variants that might affect splicing"
        )

    log["Removing variants that could affect splicing"] = count_at_different_levels(
        df.loc[df["log2FC_combined"].notnull(), :]
    )

    if make_plots:
        plot_filtering_summary(log, output_dir=f"../output/figures/{experiment_id}")

    if df.shape[0] == 0:
        print(f"WARNING: No data left after filtering {experiment_id}")
    return df


def _add_cross_reference_ids_for_interactors(df):
    """
    TODO: make clear the version numbers of all IDs

    """
    df = df.copy()
    id_map = pd.read_csv(
        "../data/internal/CCSB_ORF_ID_to_GENCODE43_Ensembl_v109_HGNC_UniProt_mapping.tsv",
        sep="\t",
    )
    if not df["orf_id_wt"].isin(id_map["orf_id"]).all():
        missing = df.loc[~df["orf_id_wt"].isin(id_map["orf_id"]), "orf_id_wt"].unique()
        print(f"WARNING: missing ORF ID mappings for: {missing}")
    if not (
        (df["interactor_id"] == EMPTY_AD_ID)
        | df["interactor_id"].isin(id_map["orf_id"])
    ).all():
        missing = df.loc[
            (df["interactor_id"] != EMPTY_AD_ID)
            & ~df["interactor_id"].isin(id_map["orf_id"]),
            "interactor_id",
        ].unique()
        print(f"WARNING: missing ORF ID mappings for: {missing}")

    df = pd.merge(
        df,
        id_map.rename(columns={c: c + "_interactor" for c in id_map.columns}),
        how="left",
        left_on="interactor_id",
        right_on="orf_id_interactor",
    )

    if (
        df.loc[df["interactor_id"] != EMPTY_AD_ID, "uniprot_ac_interactor"]
        .isnull()
        .any()
    ):
        missing = df.loc[
            df["uniprot_ac_interactor"].isnull() & (df["interactor_id"] != EMPTY_AD_ID),
            "interactor_symbol",
        ].unique()
        print(f"WARNING: missing uniprot ACs for interactors: {missing}")

    df["uniprot_ac_interactor"] = df["uniprot_ac_interactor"].astype("string")
    cols = ["uniprot_ac", "uniprot_ac_interactor"]
    df["uniprot_pair"] = (df[cols].min(axis=1) + "_" + df[cols].max(axis=1)).where(
        df[cols].notna().all(axis=1)
    )

    return df


def calc_stats_combined_across_replicates(row, n_rep=3, ref="WT"):
    """

    WARNING:
        this uses statsmodels combine_effects, which might change, so make
        sure you have the correct version installed

    TODO
    - check n replicates
    """
    if ref == "WT":
        ref_suffix = ""
    else:
        ref_suffix = "_" + ref
    out = {}
    log2fc = row[[f"log2FC{ref_suffix}_{i}" for i in range(1, n_rep + 1)]].to_numpy(
        dtype=np.float64
    )
    var = (
        row[[f"error_log2FC{ref_suffix}_{i}" for i in range(1, n_rep + 1)]].to_numpy(
            dtype=np.float64
        )
        ** 2
    )
    log2fc = log2fc[~np.isnan(log2fc)]
    var = var[~np.isnan(var)]
    if len(log2fc) == 0:
        return pd.Series(
            {
                f"log2FC{ref_suffix}_combined": np.nan,
                f"error_log2FC{ref_suffix}_combined": np.nan,
            }
        )
    elif len(log2fc) == 1:
        return pd.Series(
            {
                f"log2FC{ref_suffix}_combined": np.nan,
                f"error_log2FC{ref_suffix}_combined": np.nan,
            }
        )

    res = combine_effects(effect=log2fc, variance=var)
    out[f"log2FC{ref_suffix}_combined"] = res.mean_effect_re
    # TODO: add check for specific statsmodels version?
    out[f"error_log2FC{ref_suffix}_combined"] = np.sqrt(res.var_eff_w_re)

    return pd.Series(out)


def per_ppi_between_rep_variance_floor(df, ref_suffix=""):
    """
    Returns the between-replicate variance (tau2) for a given PPI
    combine with the per-replicate variance to get the error floor
    """
    df = df.copy()
    n_rep = _get_n_replicates(df)
    rep_cols = [f"log2FC{ref_suffix}_{i}" for i in range(1, n_rep + 1)]
    rep_err_cols = [f"error_log2FC{ref_suffix}_{i}" for i in range(1, n_rep + 1)]
    mean_centered_cols = [
        f"log2FC{ref_suffix}_{i}_mean_centered" for i in range(1, n_rep + 1)
    ]

    # need at least 2 replicates per variant
    df = df.loc[df[rep_cols].notnull().sum(axis=1) >= 2, :]
    if df.shape[0] == 0:
        return np.nan

    df[mean_centered_cols] = df[rep_cols].apply(lambda x: x - x.mean(), axis=1)
    log2fc = df[mean_centered_cols].to_numpy(dtype=np.float64).flatten()
    var = (df[rep_err_cols].to_numpy(dtype=np.float64) ** 2).flatten()
    log2fc = log2fc[~np.isnan(log2fc)]
    var = var[~np.isnan(var)]
    if len(log2fc) < 2:
        return np.nan

    res = combine_effects(
        effect=log2fc,
        variance=var,
    )
    return res.tau2


def apply_floored_error(df, ref_suffix=""):
    df = df.copy()
    n_rep = _get_n_replicates(df)
    group_cols = ["experiment", "symbol", "pool_id", "interactor_symbol"]
    tau2 = (
        df.groupby(group_cols)
        .apply(
            per_ppi_between_rep_variance_floor,
            include_groups=False,
            ref_suffix=ref_suffix,
        )
        .reset_index(name="pooled_tau2")
    )
    df = pd.merge(df, tau2, on=group_cols, how="left")
    rep_error_cols = [f"error_log2FC{ref_suffix}_{i}" for i in range(1, n_rep + 1)]
    df["error_log2FC_floor"] = (
        (df[rep_error_cols].pow(2).sum(axis=1) / n_rep) + df["pooled_tau2"]
    ).pow(0.5)

    if (
        df["error_log2FC_combined"].notnull() & df["error_log2FC_floor"].isnull()
    ).any():
        print(
            df.loc[
                df["error_log2FC_combined"].notnull()
                & df["error_log2FC_floor"].isnull(),
                :,
            ]
        )
        raise UserWarning("Something is wrong with error calculations")

    df["error_log2FC_combined"] = df[
        ["error_log2FC_combined", "error_log2FC_floor"]
    ].max(axis=1, skipna=False)
    df = df.drop(columns=["pooled_tau2", "error_log2FC_floor"])
    return df


def _add_perturbation_LLR(df):
    # hiding import to avoid circular dependency
    from perturbation_model import fit_perturbation_model, perturbation_LLR

    df = df.copy()

    model_params_path = Path("../output/fitted_mixture_model_params.json")
    if not model_params_path.exists():
        fit_perturbation_model(model_params_path)

    with open(model_params_path) as f:
        fit_params = json.load(f)

    # NOTE: this is a bit dangerous since it relies on the order of parameters
    # in the JSON file matching the order expected by the perturbation_LLR function
    df["perturbation_LLR"] = perturbation_LLR(
        df["log2FC_combined"].values,
        df["error_log2FC_combined"].values,
        list(fit_params.values()),
    )

    return df


def perturbation_status(llr, llr_threshold=2):
    if pd.isnull(llr):
        return np.nan
    if llr <= -llr_threshold:
        return "perturbed"
    elif llr >= llr_threshold:
        return "unperturbed"
    else:
        return "uncertain"


def summary_numbers_for_nanopore_Y2H_experiment(df, MIN_READS_IN_LW=MIN_READS_IN_LW):
    """

    TODO:
    - deal with empty-AD
    - number / % of variants combos under 10/100/250
    - PPIs that fail WT

    """
    df = df.copy()
    n_replicates = _get_n_replicates(df)

    tbl = {"Experiment": df["experiment"].iloc[0]}
    tbl["Number of genes"] = df["symbol"].nunique()
    tbl["Number of variant-level interactions profiled"] = df.drop_duplicates(
        subset=["symbol", "var_id", "interactor_symbol"]
    ).shape[0]

    tbl["Number of gene-level interactions profiled"] = df.drop_duplicates(
        subset=["symbol", "interactor_symbol"]
    ).shape[0]

    # TODO: inc and not inc WT
    n_var_per_gene = df.groupby("symbol")["var_id"].nunique()
    tbl["Total variants across all genes"] = n_var_per_gene.sum()
    tbl["Variants per gene"] = f"{n_var_per_gene.min()} - {n_var_per_gene.max()}"
    n_pools_per_gene = df.groupby("symbol")["pool_id"].nunique()
    tbl["Variant pools per gene"] = (
        f"{n_pools_per_gene.min()} - {n_pools_per_gene.max()}"
    )

    n_interactors_per_gene = df.groupby("symbol")["interactor_symbol"].nunique()
    # TODO: check for empty-AD
    tbl["Total interactors across all genes"] = n_interactors_per_gene.sum()
    tbl["Interactors per gene"] = (
        f"{n_interactors_per_gene.min()} - {n_interactors_per_gene.max()}"
    )

    # TODO: don't hardcode replicates
    tbl["Total -LW mapped reads"] = (
        df[["read_cnt_lw1", "read_cnt_lw2", "read_cnt_lw3"]].sum().sum()
    )
    tbl["Total +3AT mapped reads"] = (
        df[["read_cnt_3at1", "read_cnt_3at2", "read_cnt_3at3"]].sum().sum()
    )

    n_LW_measurements = df.shape[0] * n_replicates
    tbl["Total -LW measurements"] = n_LW_measurements

    # TODO: don't hardcode replicates
    tbl["Mean -LW mapped reads per variant"] = df[
        ["read_cnt_lw1", "read_cnt_lw2", "read_cnt_lw3"]
    ].mean(axis=None)
    tbl["Median -LW mapped reads per variant"] = df[
        ["read_cnt_lw1", "read_cnt_lw2", "read_cnt_lw3"]
    ].median(axis=None)

    for cutoff in [10, 100, 250]:
        tbl[f"% of -LW measurements >= {cutoff} reads"] = (
            df["read_cnt_lw1"] >= cutoff
        ).mean() * 100

    fails_any = (
        (df["read_cnt_lw1"] < MIN_READS_IN_LW)
        | (df["read_cnt_lw2"] < MIN_READS_IN_LW)
        | (df["read_cnt_lw3"] < MIN_READS_IN_LW)
    )
    tbl[f"% of variants < {MIN_READS_IN_LW} reads in at least one -LW replicate"] = (
        fails_any.mean() * 100
    )
    fails_all = (
        (df["read_cnt_lw1"] < MIN_READS_IN_LW)
        & (df["read_cnt_lw2"] < MIN_READS_IN_LW)
        & (df["read_cnt_lw3"] < MIN_READS_IN_LW)
    )
    tbl[f"% of variants < {MIN_READS_IN_LW} reads in all -LW replicates"] = (
        fails_all.mean() * 100
    )
    return pd.Series(tbl)


def _merge_values(group):
    """
    Uses a random effects model to combine log2FC values
    TODO: this is duplicated code with the replicates combining code
    """
    out = group.iloc[0].copy()
    # WARNING: this is a bit dangerous, e.g. log2FC could show up in a future
    # variable we want to keep
    ptn = r"read_cnt|total_read|log2FC|growth_score|pct_well_to_mapped|^pool_id$|^p_val_combined$|^z_score$|^perturbation_LLR$|^perturbation_status$"
    out[[c for c in group.columns if bool(re.search(ptn, c))]] = np.nan
    out["experiment"] = "merge:" + "|".join(group["experiment"].unique())
    out["is_merge_within_experiment"] = group["is_merge_within_experiment"].any()
    out["is_measured_multiple_within_experiment"] = group[
        "is_measured_multiple_within_experiment"
    ].any()

    log2fc = group["log2FC_combined"].values
    var = group["error_log2FC_combined"].values ** 2
    log2fc = log2fc[~np.isnan(log2fc)]
    var = var[~np.isnan(var)]
    if len(log2fc) < 2:
        raise ValueError("Should only be merging groups with multiple values")

    res = combine_effects(effect=log2fc, variance=var)
    out["log2FC_combined"] = res.mean_effect_re
    out["error_log2FC_combined"] = np.sqrt(res.var_eff_w_re)

    if group["experiment"].nunique() > 1:
        out["is_merge_across_experiments"] = True
    else:
        out["is_merge_within_experiment"] = True
    return out


def combine_results_across_experiments(df, drop_superceded=True):
    """
    For each variant PPI that appears more than once in an experiment
    add a line to the dataframe with the combined stats
    NOTE: assuming same ORF used
       TODO: check this
    NOTE: assuming same nt variant per a.a. variant
        TODO: check this (if not true, does it matter?)
    """
    df = df.copy()

    df["is_measured_multiple_within_experiment"] = (
        df.groupby(["experiment", "symbol", "interactor_symbol", "var_id"])[
            "log2FC_combined"
        ].transform(lambda x: x.notnull().sum())
        >= 2
    )
    df["is_measured_in_multiple_experiments"] = (
        df.set_index(["symbol", "interactor_symbol", "var_id"])
        .index.map(
            df.loc[df["log2FC_combined"].notnull()]
            .groupby(["symbol", "interactor_symbol", "var_id"])["experiment"]
            .nunique()
            >= 2
        )
        .fillna(False)
        .astype("bool")
    )

    df["is_merge_within_experiment"] = False
    df["is_merge_across_experiments"] = False
    df["is_replaced_by_merged_stat"] = False
    merges_within = (
        df.loc[
            (df["var_id"] != "WT")
            & df["log2FC_combined"].notnull()
            & df["is_measured_multiple_within_experiment"],
            :,
        ]
        .groupby(["experiment", "symbol", "interactor_symbol", "var_id"])[df.columns]
        .apply(_merge_values)
    )
    df = pd.concat([df, merges_within], ignore_index=True)
    df.loc[
        df["is_measured_multiple_within_experiment"]
        & ~df["is_merge_within_experiment"],
        "is_replaced_by_merged_stat",
    ] = True
    merges_between = (
        df.loc[
            (df["var_id"] != "WT")
            & df["log2FC_combined"].notnull()
            & ~df["is_replaced_by_merged_stat"]
            & df["is_measured_in_multiple_experiments"],
            :,
        ]
        .groupby(["symbol", "interactor_symbol", "var_id"])[df.columns]
        .apply(_merge_values)
    )
    df = pd.concat([df, merges_between], ignore_index=True)

    df.loc[
        df["is_measured_in_multiple_experiments"] & ~df["is_merge_across_experiments"],
        "is_replaced_by_merged_stat",
    ] = True

    df["z_score"] = df["log2FC_combined"] / df["error_log2FC_combined"]
    if "perturbation_LLR" in df.columns:
        df = _add_perturbation_LLR(df)
        df["perturbation_status"] = df["perturbation_LLR"].apply(perturbation_status)

    if drop_superceded:
        df = df.loc[~df["is_replaced_by_merged_stat"], :]
        nulls_not_merged = (
            (df["var_id"] != "WT")
            & df.duplicated(
                subset=["symbol", "var_id", "interactor_symbol"], keep=False
            )
            & df["log2FC_combined"].isnull()
        )
        df = df.loc[~nulls_not_merged, :]

    return df


def edgotype_status(status_list):
    status_list = status_list.to_list()
    if len(status_list) == 1:
        return np.nan
    if pd.isnull(status_list).all():
        return np.nan
    if "perturbed" in status_list and "unperturbed" in status_list:
        return "edgetic"
    elif status_list.count("perturbed") >= 2:
        return "quasi-null"
    elif status_list.count("unperturbed") >= 2:
        return "quasi-wildtype"
    else:
        return "uncertain"


def per_variant_edgotype_table(df):

    edgotype = (
        df.loc[df["var_id"] != "WT", :]
        .groupby(["symbol", "var_id"])["perturbation_status"]
        .apply(edgotype_status)
    )
    edgotype = edgotype.to_frame().rename(
        columns={"perturbation_status": "edgotype_status"}
    )
    edgotype["n_ppi"] = (
        df.loc[df["var_id"] != "WT", :]
        .groupby(["symbol", "var_id"])["interactor_symbol"]
        .nunique()
    )
    edgotype["n_measured_ppi"] = (
        df.loc[df["var_id"] != "WT", :]
        .groupby(["symbol", "var_id"])["log2FC_combined"]
        .apply(lambda x: x.notnull().sum())
    )

    for status in ["perturbed", "unperturbed", "uncertain"]:
        edgotype[f"n_{status}"] = (
            df.loc[df["var_id"] != "WT", :]
            .groupby(["symbol", "var_id"])["perturbation_status"]
            .apply(lambda x: (x == status).sum())
        )

    edgotype["edgetic_confidence"] = (
        df.loc[df["var_id"] != "WT", :]
        .groupby(["symbol", "var_id"])["perturbation_LLR"]
        .apply(lambda x: max([min([x.max(), -x.min()]), 0]))
    )
    llr_threshold = 2  # TODO read from config
    if (
        edgotype.loc[
            edgotype["edgotype_status"] == "edgetic", "edgetic_confidence"
        ].min()
        < llr_threshold
    ):
        raise UserWarning(
            f"Edgetic confidence score less than threshold of {llr_threshold} found"
        )
    if (
        edgotype.loc[
            edgotype["edgotype_status"] != "edgetic", "edgetic_confidence"
        ].max()
        > llr_threshold
    ):
        raise UserWarning(
            f"Non-edgetic confidence score greater than threshold of {llr_threshold} found"
        )

    edgotype = edgotype.reset_index()

    n_b4 = edgotype.shape[0]
    edgotype = pd.merge(
        edgotype,
        df.loc[
            :,
            [
                "symbol",
                "ensembl_gene_id",
                "var_id",
                "cv_clinical_significance",
                "clinical_significance_simple",
                "mondo_ids",
                "high_level_mondo_terms",
                "blosum62",
                "phyloP",
                "allele_frequency",
                "moi",
                "am_pathogenicity",
                "pLDDT",
                "RSA",
                "is_disordered",
            ],
        ].drop_duplicates(),
        on=[
            "symbol",
            "var_id",
        ],
        how="left",
    )
    if edgotype.shape[0] != n_b4:
        # TODO: fix this and change print to UserWarning
        print(
            "WARNING: problem with merge. Probably multiple nt variants per aa variant"
        )

    n_vars_no_wt = (
        df.loc[df["var_id"] != "WT", ["symbol", "var_id"]].drop_duplicates().shape[0]
    )
    if edgotype.shape[0] != n_vars_no_wt:
        print(
            f"WARNING: problem with edgotype table vs original table. {edgotype.shape[0]} vs {n_vars_no_wt} rows"
        )

    return edgotype
