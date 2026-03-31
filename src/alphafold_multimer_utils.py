from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser


def read_contact_probability(in_dir):
    """
    TODO: the code to make these files just works for AF2.
    AF3 has some values called contact probability but not
    sure if they are the same.

    """
    in_dir = Path(in_dir)
    data = []
    for fpath in in_dir.glob("*/ranked_0_contact_probability.txt"):
        with open(fpath, "r") as f:
            file_contents = f.read()
            if file_contents == "":
                cp = np.nan
                warnings.warn(f"empty file: {fpath}")
            else:
                cp = float(file_contents)
            data.append((fpath.parents[0].stem, cp))
    return pd.DataFrame(data=data, columns=["pair", "contact_probability"])


def read_ipae(in_dir):
    in_dir = Path(in_dir)
    data = []
    for fpath in in_dir.glob("*/ranked_0_iPAE.txt"):
        with open(fpath, "r") as f:
            data.append((fpath.parents[0].stem, float(f.read())))
    return pd.DataFrame(data=data, columns=["pair", "iPAE"])


def read_iptm_plus_ptm(in_dir):
    in_dir = Path(in_dir)
    data = []
    for fpath in in_dir.glob("*/ranked_0_iPTM-plus-PTM.txt"):
        with open(fpath, "r") as f:
            data.append((fpath.parents[0].stem, float(f.read())))
    return pd.DataFrame(data=data, columns=["pair", "iPTM+PTM"])


def read_iptm_and_ptm(in_dir):
    """
    AF3 only
    """
    in_dir = Path(in_dir)
    data = []
    for fpath in in_dir.glob("*/*_summary_confidences_0.json"):
        with open(fpath, "r") as f:
            conf = json.load(f)
            data.append(
                (
                    fpath.parents[0].stem,
                    float(conf["iptm"]),
                    float(conf["ptm"]),
                )
            )
    return pd.DataFrame(data=data, columns=["pair", "iPTM", "PTM"])


def read_pdockq(in_dir):
    in_dir = Path(in_dir)
    data = []
    for fpath in in_dir.glob("*/ranked_0_pdockq.out"):
        with open(fpath, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            data.append([fpath.parents[0].stem, np.nan, np.nan])
            continue
        data.append(
            (
                lines[0].split()[-1].split("/")[-2],
                float(lines[0].split()[2]),
                float(lines[1].split()[-1]),
            )
        )
    return pd.DataFrame(data=data, columns=["pair", "pDockQ", "PPV_from_pDockQ"])


def read_alphafold_dimer_metrics(in_dir, prefix=None):
    in_dir = Path(in_dir)
    df = pd.merge(
        read_contact_probability(in_dir), read_pdockq(in_dir), on="pair", how="outer"
    )
    df = pd.merge(df, read_iptm_and_ptm(in_dir), on="pair", how="outer")
    df = pd.merge(df, read_ipae(in_dir), on="pair", how="outer")

    _full_contacts, summary_contacts = load_alphafold_residue_contacts(in_dir)
    if summary_contacts["pair"].duplicated().any():
        raise UserWarning("unexpected duplicates")
    df["n_contacts_PAE_lt_4A"] = df["pair"].map(
        summary_contacts.set_index("pair")["Total_interactions_PAE_lt_4A"]
    )

    if prefix is not None:
        df = df.rename(columns={c: prefix + "_" + c for c in df.columns if c != "pair"})
    df["gene_name_a"] = df["pair"].apply(lambda x: x.split("_")[0])
    df["gene_name_b"] = df["pair"].apply(lambda x: x.split("_")[1])
    first_cols = ["pair", "gene_name_a", "gene_name_b"]
    df = df.loc[:, first_cols + list([c for c in df.columns if c not in first_cols])]

    if df.isnull().any().any():
        print("Warning: some missing values in alphafold metrics")

    return df


def get_lengths_from_PDB_file(pdb_file_path):
    if pdb_file_path.suffix == ".pdb":
        parser = PDBParser()
    elif pdb_file_path.suffix == ".cif":
        parser = MMCIFParser()
    structure = parser.get_structure("tmp", pdb_file_path)
    return [len(list(c.get_residues())) for c in structure.get_chains()]


def read_residue_contacts(in_path):
    with open(in_path, "r") as f:
        lines = f.readlines()
    if lines[0] != "    Residue     Atom   Residue     Atom  Type     Dist\n":
        raise UserWarning("unexpected format")
    table_per_residue = [l.strip().split() for l in lines[:-7]]
    table_per_residue = pd.DataFrame(
        columns=[
            "Residue_X",
            "Residue_index_X",
            "Atom_X",
            "Residue_Y",
            "Residue_index_Y",
            "Atom_Y",
            "Type",
            "Dist",
        ],
        data=table_per_residue[1:] if len(table_per_residue) > 1 else [],
    )
    table_summary = [(l.split(":")[0], int(l.split()[-1].strip())) for l in lines[-6:]]
    table_summary = pd.DataFrame(data=table_summary).set_index(0).T
    table_per_residue["Residue_index_X"] = table_per_residue["Residue_index_X"].astype(
        int
    )
    table_per_residue["Residue_index_Y"] = table_per_residue["Residue_index_Y"].astype(
        int
    )
    return (table_per_residue, table_summary)


def _find_pdb_file_path(in_dir):
    af2_file = in_dir / "ranked_0.pdb"
    if af2_file.exists():
        return af2_file
    af3_files = list(in_dir.glob("*_model.cif"))
    if len(af3_files) > 1:
        raise UserWarning(f"Not sure which file: {af3_files}")
    elif len(af3_files) == 1:
        return af3_files[0]
    all_pdb_files = list(in_dir.glob("*.pdb"))
    if len(all_pdb_files) > 1:
        raise UserWarning(f"Not sure which file: {af3_files}")
    elif len(all_pdb_files) == 1:
        return all_pdb_files[0]
    elif len(all_pdb_files) == 0:
        raise UserWarning(f"No .pdb file in {in_dir}")


def _load_pae(in_dir):
    af2_file = in_dir / "ranked_0_PAE.csv"
    if af2_file.exists():
        pae = pd.read_csv(af2_file, index_col=0)
        pae.columns = pae.columns.astype(int)
        return pae
    af3_files = list(in_dir.glob("*_full_data_0.json"))
    if len(af3_files) == 0:
        raise UserWarning(f"couldn't find pae data in {in_dir}")
    elif len(af3_files) > 1:
        raise UserWarning(f"multiple files matching *_full_data_0.json in {in_dir}")
    with open(af3_files[0], "r") as f:
        pae = json.load(f)["pae"]
        pae = pd.DataFrame(
            data=pae,
            columns=range(len(pae)),
            index=range(len(pae)),
        )
    return pae


def load_alphafold_residue_contacts(in_dir):
    """
    TODO: this is a bit slow for large datasets. Do a line-by-line time profiling.
    """
    in_dir = Path(in_dir)
    full_table = []
    summary_table = []
    for fpath in in_dir.glob("*/ranked_0_residue_contacts_Interactome3D.out"):
        pair = fpath.parents[0].stem
        gene_name_a, gene_name_b = pair.split("_")
        len_a, len_b = get_lengths_from_PDB_file(_find_pdb_file_path(fpath.parent))
        df_a, df_b = read_residue_contacts(fpath)
        # NOTE: this is 0-indexed and the two proteins are concatenated
        pae = _load_pae(fpath.parents[0])
        if df_a.shape[0] > 0:
            df_a["PAE"] = df_a.apply(
                lambda row: pae.at[
                    row["Residue_index_X"] - 1, (len_a + row["Residue_index_Y"]) - 1
                ],
                axis=1,
            )
        else:
            df_a["PAE"] = None
        df_a["pair"] = pair
        df_a["Chain_X"] = "A"
        df_a["Chain_Y"] = "B"
        df_b["pair"] = pair
        df_b["Chain_X"] = "A"
        df_b["Chain_Y"] = "B"
        if df_a.shape[0] > 0:
            full_table.append(df_a.copy())
        summary_table.append(df_b.copy())
    full_table = pd.concat(full_table)
    summary_table = pd.concat(summary_table)

    full_table["gene_name_a"] = full_table["pair"].apply(lambda x: x.split("_")[0])
    full_table["gene_name_b"] = full_table["pair"].apply(lambda x: x.split("_")[1])
    summary_table["gene_name_a"] = summary_table["pair"].apply(
        lambda x: x.split("_")[0]
    )
    summary_table["gene_name_b"] = summary_table["pair"].apply(
        lambda x: x.split("_")[1]
    )

    full_table = full_table.loc[
        :, list(full_table.columns[-5:]) + list(full_table.columns[:-5])
    ]
    summary_table = summary_table.loc[
        :, list(summary_table.columns[-5:]) + list(summary_table.columns[:-5])
    ]

    # The 4 Angstrom cut was an arbitrary choice
    summary_table["Total_interactions_PAE_lt_4A"] = (
        summary_table["pair"]
        .map(full_table.loc[full_table["PAE"] < 4, :].groupby("pair").size())
        .fillna(0)
        .astype(int)
    )

    return full_table, summary_table


def load_probable_residue_contacts(in_dir, cutoff=0.95):
    in_dir = Path(in_dir)

    # NOTE: sorting so that the most recent results are used
    # i.e. AF3 appends the date to folder name with multiple runs
    # so files can look like: q9umx0_q9y5u9_20251020_095156/q9umx0_q9y5u9_confidences.json
    latest = {}
    for fpath in sorted(in_dir.glob("*/*_confidences.json")):
        if fpath.stem.endswith("summary_confidences"):
            continue
        pair_id = fpath.stem.replace("_confidences", "")
        latest[pair_id] = fpath

    rrc = []
    for pair_id, fpath in latest.items():
        with open(fpath, "r") as f:
            conf = json.load(f)
        all_contact_probs = np.asarray(conf["contact_probs"])
        len_a = conf["token_chain_ids"].index("B")
        contact_probs = all_contact_probs[:len_a, len_a:]
        ix, iy = np.where(contact_probs > cutoff)
        probs = contact_probs[ix, iy]
        pairs = pd.DataFrame(
            {"Residue_index_X": ix, "Residue_index_Y": iy, "contact_probability": probs}
        )
        pairs["pair"] = pair_id

        rrc.append(pairs)

    if not rrc:
        return pd.DataFrame(
            columns=[
                "Residue_index_X",
                "Residue_index_Y",
                "contact_probability",
                "pair",
                "gene_name_a",
                "gene_name_b",
            ]
        )
    rrc = pd.concat(rrc, ignore_index=True)
    rrc["Residue_index_X"] += 1
    rrc["Residue_index_Y"] += 1
    genes = rrc["pair"].str.split("_", n=2, expand=True)
    rrc["gene_name_a"] = genes[0]
    rrc["gene_name_b"] = genes[1]

    return rrc
