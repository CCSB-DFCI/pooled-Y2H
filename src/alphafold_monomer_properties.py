"""
TODO: cache (joblib memory?)
"""

import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

from Bio.PDB import MMCIFParser, DSSP


def alphafold_monomer_residue_properties(uniprot_ac):
    """
    TODO:
        - add residue type

    """
    file_name = f"AF-{uniprot_ac}-F1-model_v6.cif"
    file_url = f"https://alphafold.ebi.ac.uk/files/{file_name}"
    output_dir = Path("../data/external/alphafold_monomers")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    if not output_path.exists():
        try:
            urlretrieve(file_url, output_path)
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")
            return pd.DataFrame(
                columns=[
                    "uniprot_ac",
                    "position",
                    "residue",
                    "pLDDT",
                    "RSA",
                    "secondary_structure",
                    "RSA_window_20",
                    "is_disordered",
                ]
            )

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(file_name.split(".")[0], output_path)
    plddt = [r["CA"].bfactor for r in structure.get_residues() if "CA" in r]

    dssp = DSSP(structure[0], output_path, acc_array="Wilke")
    # RSA is the 4th column of output
    rsa = [x[3] for x in dssp]

    # disorder
    df = pd.DataFrame(
        data={
            "uniprot_ac": uniprot_ac,
            "position": list(range(1, len(plddt) + 1)),
            "residue": [
                x[1] for x in dssp
            ],  # TODO check if I guessed the correct index
            "pLDDT": plddt,
            "RSA": rsa,
            "secondary_structure": [x[2] for x in dssp],
        }
    )
    if (df["RSA"] > 1.0).any():
        raise UserWarning("Expected RSA to be clipped to be below 1.0")
    WINDOW_SIZE_RESIDUES = 20
    DISORDER_WINDOW_RSA_CUTOFF = 0.5
    rsa_window_col = f"RSA_window_{WINDOW_SIZE_RESIDUES}"
    df[rsa_window_col] = (
        df["RSA"]
        .rolling(
            window=WINDOW_SIZE_RESIDUES * 2 + 1,
            min_periods=WINDOW_SIZE_RESIDUES + 1,
            center=True,
        )
        .mean()
        .rename(rsa_window_col)
    )
    df["is_disordered"] = df[rsa_window_col] >= DISORDER_WINDOW_RSA_CUTOFF

    # correct for long helices which are structured, usually bound to a partner
    # but have high RSA in the monomer state
    DISORDER_HELIX_LENGTH_CUTOFF = 20
    to_change = []
    helix_count = 0
    for _i, row in df.iterrows():
        if row["secondary_structure"] == "H":
            helix_count += 1
        else:
            if helix_count >= DISORDER_HELIX_LENGTH_CUTOFF:
                for i in range(row["position"] - 1, row["position"] - helix_count, -1):
                    to_change.append(i)
            helix_count = 0
    # catch helix at end of sequence
    if helix_count >= DISORDER_HELIX_LENGTH_CUTOFF:
        for i in range(row["position"], row["position"] - helix_count, -1):
            to_change.append(i)
    df.loc[df["position"].isin(to_change), "is_disordered"] = False

    return df


def add_alphafold_monomer_properties(df):
    """ "
    TODO:
    - check ORF sequence against uniprot sequence
    - drop the added 'residue' column or rename to residue in af structure
    """
    df["uniprot_ac_if_canonical_else_iso"] = df["uniprot_isoform_ac"]
    df.loc[
        df["wt_orf_matches_uniprot_canonical"], "uniprot_ac_if_canonical_else_iso"
    ] = df.loc[df["wt_orf_matches_uniprot_canonical"], "uniprot_ac"]

    monomer_props = pd.concat(
        [
            alphafold_monomer_residue_properties(uniprot_ac)
            for uniprot_ac in df["uniprot_ac_if_canonical_else_iso"].unique()
        ]
    )

    n_rows_b4 = df.shape[0]
    df = pd.merge(
        df,
        monomer_props.rename(
            columns={
                "uniprot_ac": "uniprot_ac_if_canonical_else_iso",
                "position": "aa_pos_uniprot_isoform",
                "residue": "residue_in_af_structure",
            }
        ),
        how="left",
        on=["uniprot_ac_if_canonical_else_iso", "aa_pos_uniprot_isoform"],
    )
    if df.shape[0] != n_rows_b4:
        raise UserWarning(
            "Merging monomer properties changed number of rows, something went wrong"
        )

    df.loc[
        df["pct_identity_with_uniprot_sequence"] < 95,
        ["pLDDT", "RSA", "secondary_structure", "RSA_window_20", "is_disordered"],
    ] = pd.NA

    return df
