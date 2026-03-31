from pathlib import Path
import json

import numpy as np
from Bio.PDB import MMCIFParser

from alphafold_multimer_utils import load_probable_residue_contacts

# TODO move to data/internal
AF3_DIMER_MODELS_PATH = Path(
    "/Users/lukelambourne/Dropbox (Partners HealthCare)/VarChamp/predicted_structures"
)


# TODO: cache
def load_intrachain_pae(pair, chain_id):
    in_path = AF3_DIMER_MODELS_PATH
    if chain_id not in ["A", "B"]:
        raise ValueError("chain_id must be 'A' or 'B'")
    file_path = list(
        sorted(in_path.glob(f"{pair.lower()}*/{pair.lower()}_confidences.json"))
    )[-1]
    with open(file_path, "r") as f:
        conf = json.load(f)
    len_a = conf["token_chain_ids"].index("B")
    pae = np.array(conf["pae"])
    if chain_id == "A":
        pae = pae[0:len_a, 0:len_a]
    else:  # chain_id == 'B'
        pae = pae[len_a:, len_a:]
    return pae


def load_interface_residues_from_AF3_predictions():
    in_path = AF3_DIMER_MODELS_PATH
    rrc = load_probable_residue_contacts(in_path, cutoff=0.95)
    rrc = rrc.rename(
        columns={"gene_name_a": "uniprot_ac_a", "gene_name_b": "uniprot_ac_b"}
    )
    return rrc


def add_AF3_dimer_info(df, verbose=True):
    """
    - confident
    - is interface residue
    - distance from interface (plddt filtered (or should be PAE right?))
    """
    if "uniprot_pair" not in df.columns:
        raise ValueError("DataFrame must have 'uniprot_pair' column")
    if df["uniprot_pair"].isnull().any():
        print("WARNING: missing uniprot_pair values")
    in_path = AF3_DIMER_MODELS_PATH
    df = df.copy()
    pairs_with_model = {
        p.name.replace("_model.cif", "").upper() for p in in_path.glob("*/*_model.cif")
    }
    df["attempted_AF3_dimer_model"] = df["uniprot_pair"].isin(pairs_with_model)
    rrc = load_interface_residues_from_AF3_predictions()
    high_confidence_af3_pairs = (
        rrc.groupby("pair")
        .size()
        .loc[(rrc.groupby("pair").size() >= 5)]
        .index.map(lambda x: x.upper())
    )
    df["has_confident_af3_dimer_model"] = df["uniprot_pair"].isin(
        high_confidence_af3_pairs
    )

    df["is_interface_residue_in_af3_dimer"] = np.nan
    df["is_interface_residue_in_af3_dimer"] = df[
        "is_interface_residue_in_af3_dimer"
    ].astype("boolean")
    df["distance_from_af3_dimer_interface"] = np.nan

    for uniprot_ac, pair in (
        df.loc[df["has_confident_af3_dimer_model"], ["uniprot_ac", "uniprot_pair"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        interface_residues = _get_interface_residues(rrc, uniprot_ac, pair)
        if len(interface_residues) == 0:
            raise UserWarning("problem loading interface residues")

        selection = (df["uniprot_ac"] == uniprot_ac) & (df["uniprot_pair"] == pair)
        df.loc[selection, "is_interface_residue_in_af3_dimer"] = df.loc[
            selection, "aa_pos_uniprot_canonical"
        ].isin(interface_residues)
        df.loc[
            df["aa_pos_uniprot_canonical"].isnull(), "is_interface_residue_in_af3_dimer"
        ] = np.nan

        residue_pos = df.loc[selection, "aa_pos_uniprot_canonical"].values
        df.loc[selection, "distance_from_af3_dimer_interface"] = (
            distance_from_interface(
                chain=_load_af3_dimer_chain(uniprot_ac, pair, in_path),
                interface_residue_numbers=interface_residues,
                query_residue_numbers=residue_pos,
                atom_name="CA",
            )
        )

    if verbose:
        _print_AF3_numbers_summary(df)
    _test_AF3_info_consistency(df)
    if df["attempted_AF3_dimer_model"].all():
        df = df.drop(columns=["attempted_AF3_dimer_model"])

    return df


def distance_from_interface(
    chain,
    interface_residue_numbers,
    query_residue_numbers,
    pae_max=10.0,
    atom_name="CA",
):
    """
    TODO: test code carefully. Got from ChatGPT
    """
    pae = load_intrachain_pae(
        pair=chain.get_parent().get_parent().get_id(), chain_id=chain.id
    )

    residues = [r for r in chain if r.id[0] == " " and (atom_name in r)]
    resseq_to_i = {r.id[1]: i for i, r in enumerate(residues)}

    N = len(residues)
    if pae.shape[0] != N or pae.shape[1] != N:
        raise ValueError(
            f"PAE shape {pae.shape} does not match number of residues with {atom_name} in chain ({N}). "
            "You may need to align the chain residue list to the model indexing."
        )

    coords = np.vstack([r[atom_name].coord for r in residues])  # (N, 3)

    # Interface indices that exist in this chain
    interface_idx = np.array(
        [resseq_to_i[x] for x in interface_residue_numbers if x in resseq_to_i],
        dtype=np.int32,
    )
    if interface_idx.size == 0:
        return [np.nan] * len(query_residue_numbers)

    distances = {}
    for q in query_residue_numbers:
        qi = resseq_to_i.get(q)
        if qi is None:
            distances[q] = np.nan
            continue

        # PAE filter: keep only interface residues with confident relative placement to q
        ok = pae[qi, interface_idx] <= pae_max
        if not np.any(ok):
            distances[q] = np.nan
            continue

        idx_ok = interface_idx[ok]
        d = np.linalg.norm(coords[idx_ok] - coords[qi], axis=1)
        distances[q] = float(d.min()) if d.size else np.nan

    return [distances[resseq] for resseq in query_residue_numbers]


def _load_af3_dimer_chain(uniprot_ac, pair, in_path):
    if pair.split("_")[0] == uniprot_ac:
        chain_id = "A"
        chain_idx = 0
    elif pair.split("_")[1] == uniprot_ac:
        chain_id = "B"
        chain_idx = 1
    else:
        raise ValueError("uniprot_ac not in pair")
    cif_path = list(sorted(in_path.glob(f"{pair.lower()}*/{pair.lower()}_model.cif")))[
        -1
    ]
    parser = MMCIFParser()
    structure = parser.get_structure(pair, cif_path)
    chain = list(structure.get_chains())[chain_idx]
    if chain.id != chain_id:
        raise ValueError("Chain ID mismatch")
    return chain


def _get_interface_residues(rrc, uniprot_ac, pair):
    if pair.split("_")[0] == uniprot_ac:
        res_col = "Residue_index_X"
    elif pair.split("_")[1] == uniprot_ac:
        res_col = "Residue_index_Y"
    else:
        raise ValueError("uniprot_ac not in pair")
    interface_residues = set(rrc.loc[rrc["pair"] == pair.lower(), res_col].unique())
    return interface_residues


def _print_AF3_numbers_summary(df):
    n_total = df["uniprot_pair"].nunique()
    n_attempted = df.loc[df["attempted_AF3_dimer_model"], "uniprot_pair"].nunique()
    n_confident = df.loc[df["has_confident_af3_dimer_model"], "uniprot_pair"].nunique()
    n_tested_interface_residues = (
        df.loc[
            df["log2FC_combined"].notnull() & df["has_confident_af3_dimer_model"],
            "is_interface_residue_in_af3_dimer",
        ]
        == True
    ).sum()
    n_tested_possible_residues = (
        df.loc[
            df["log2FC_combined"].notnull() & df["has_confident_af3_dimer_model"],
            "is_interface_residue_in_af3_dimer",
        ].notnull()
    ).sum()
    n_pairs_with_interface_residues = (
        df.groupby(["uniprot_ac", "uniprot_pair"])["is_interface_residue_in_af3_dimer"]
        .apply(lambda x: (x == True).any())
        .sum()
    )
    # NOTE: this is side-dependent whereas n_condident is just one count per pair
    n_pairs_with_models = (
        df.groupby(["uniprot_ac", "uniprot_pair"])["is_interface_residue_in_af3_dimer"]
        .apply(lambda x: x.notnull().any())
        .sum()
    )
    pairs_on_both_side = (
        df.loc[df["has_confident_af3_dimer_model"]]
        .groupby("uniprot_pair")
        .filter(lambda x: x["uniprot_ac"].nunique() == 2)["uniprot_pair"]
        .unique()
    )

    print(
        f"{n_attempted} of {n_total} ({n_attempted / n_total:.0%}) pairs have AF3 dimers attempted"
    )
    print(
        f"{n_confident} of {n_attempted} ({n_confident / n_attempted:.0%}) pairs have confident AF3 dimer models"
    )
    print(
        f"{n_tested_interface_residues} of {n_tested_possible_residues} ({n_tested_interface_residues / n_tested_possible_residues:.0%}) tested variants are at interface residues in confident AF3 dimers"
    )
    print(
        f"{n_pairs_with_interface_residues} of {n_pairs_with_models} ({n_pairs_with_interface_residues / n_pairs_with_models:.0%}) pairs with confident AF3 dimer models have at least one tested variant at an interface residue"
    )
    print(
        f"{len(pairs_on_both_side)} pairs with confident AF3 dimer models where we've tested variants on both sides of the interaction:"
    )
    print(
        df.loc[
            df["uniprot_pair"].isin(pairs_on_both_side),
            ["symbol", "interactor_symbol", "uniprot_pair"],
        ]
        .drop_duplicates()
        .sort_values("uniprot_pair")
    )


def _test_AF3_info_consistency(df):
    if "attempted_AF3_dimer_model" in df.columns:
        if df.loc[
            ~df["attempted_AF3_dimer_model"], "has_confident_af3_dimer_model"
        ].any():
            raise UserWarning(
                "Rows without attempted_AF3_dimer_model should not have has_confident_af3_dimer_model=True"
            )
        if df.loc[
            ~df["attempted_AF3_dimer_model"], "is_interface_residue_in_af3_dimer"
        ].any():
            raise UserWarning(
                "Rows without attempted_AF3_dimer_model should not have is_interface_residue_in_af3_dimer=True"
            )
        if (
            df.loc[
                ~df["attempted_AF3_dimer_model"], "distance_from_af3_dimer_interface"
            ]
            .notnull()
            .any()
        ):
            raise UserWarning(
                "Rows without attempted_AF3_dimer_model should not have distance_from_af3_dimer_interface values"
            )
    if df.loc[
        df["has_confident_af3_dimer_model"] == False,
        "is_interface_residue_in_af3_dimer",
    ].any():
        raise UserWarning(
            "Rows without has_confident_af3_dimer_model should not have is_interface_residue_in_af3_dimer=True"
        )
    if (
        df.loc[
            df["has_confident_af3_dimer_model"] == False,
            "distance_from_af3_dimer_interface",
        ]
        .notnull()
        .any()
    ):
        raise UserWarning(
            "Rows without has_confident_af3_dimer_model should not have distance_from_af3_dimer_interface values"
        )
    if not (
        df.loc[
            df["is_interface_residue_in_af3_dimer"] == True,
            "distance_from_af3_dimer_interface",
        ]
        == 0
    ).all():
        raise UserWarning(
            "Rows with is_interface_residue_in_af3_dimer=True should have distance_from_af3_dimer_interface of 0"
        )
