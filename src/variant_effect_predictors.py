from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data import IUPACData


def load_alphamissense_scores(uniprot_acs=None):
    am = _load_all_alphamissense_scores()
    if uniprot_acs is None:
        return am
    am = am.loc[am["uniprot_id"].isin(uniprot_acs), :].copy()
    return am


def _load_all_alphamissense_scores():

    cache_dir = Path("../cache/")
    # File is large, so cache as parquet
    cached_file = cache_dir / "alphamissense_scores.parquet"
    if cached_file.exists():
        am = pd.read_parquet(cached_file)
        return am

    am = pd.read_csv(
        "../data/external/AlphaMissense_aa_substitutions.tsv", sep="\t", skiprows=3
    )
    one_to_three = IUPACData.protein_letters_1to3
    am["aa_pos"] = am["protein_variant"].str.slice(1, -1).astype(int)
    am["variant_aa_1letter"] = am["protein_variant"].str[-1]
    am["variant_aa_3letter"] = am["variant_aa_1letter"].map(one_to_three)
    am.to_parquet(cached_file, index=False)
    return am


def add_alphamissense_column(df):
    """
    NOTE: these are based on UniProt canonical sequences
    """
    df = df.copy()
    am = load_alphamissense_scores(uniprot_acs=df["uniprot_ac"].unique())
    three_to_one = IUPACData.protein_letters_3to1
    df["aa_change_one_letter"] = df["aa_change_uniprot_canonical"].apply(
        lambda x: (
            "".join([three_to_one[x[:3]], x[3:-3], three_to_one[x[-3:]]])
            if pd.notnull(x)
            else np.nan
        )
    )
    df = pd.merge(
        df,
        am.rename(
            columns={
                "uniprot_id": "uniprot_ac",
                "protein_variant": "aa_change_one_letter",
            }
        ).loc[
            :, ["uniprot_ac", "aa_change_one_letter", "am_pathogenicity", "am_class"]
        ],
        how="left",
        on=["uniprot_ac", "aa_change_one_letter"],
    )

    df.loc[
        df["pct_identity_with_uniprot_sequence"] < 95,
        ["am_pathogenicity", "am_class"],
    ] = pd.NA

    return df
