import re


def _get_n_replicates(df):

    if "read_cnt_3at1" in df.columns:
        selection_read_col_prefix = "read_cnt_3at"
    elif "read_cnt_lw1" in df.columns:
        selection_read_col_prefix = "read_cnt_lw"
    elif "read_cnt_qc1" in df.columns:
        selection_read_col_prefix = "read_cnt_qc"
    else:
        raise ValueError("did not have expected read count column names")

    selection_read_cols = [
        c for c in df.columns if c.startswith(selection_read_col_prefix)
    ]
    rep_numbers = list(
        sorted(
            [int(c.replace(selection_read_col_prefix, "")) for c in selection_read_cols]
        )
    )
    if rep_numbers != list(range(1, rep_numbers[-1] + 1)):
        raise UserWarning("replicate numbers are not consecutive from 1 to n")
    return len(rep_numbers)


def sort_var(s):
    nt_change_pattern = re.compile(r"(\d+)[A-Z]\>([A-Z])")
    aa_change_pattern = re.compile(r"[A-Za-z]{3}(\d+)([A-Za-z]{3})")
    if s == "WT":
        return 0
    elif nt_change_pattern.match(s):
        return sort_nt_var(s)
    elif aa_change_pattern.match(s):
        return sort_aa_var(s)
    else:
        print("Error parsing variant:", s)
        return 0


def sort_nt_var(s):
    if s == "WT":
        return 0
    else:
        return int(s[:-3])


def sort_aa_var(s):
    """
    Options:
    - Three letter a.a. change: like "Cys39Ala"
    - "WT"
    - multiple mutations separated by "_": "Cys39Ala,Ser40Gly"
    """
    if s == "WT":
        return 0
    else:
        try:
            return int(s.split("_")[0][3:-3]) + len(s.split("_")) * 1e-3
        except:
            print("Error parsing amino acid variant:", s)
            return 0
