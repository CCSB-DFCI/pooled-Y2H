from pathlib import Path
from multiprocessing import Pool
from functools import reduce, lru_cache
import time

import numpy as np
import pandas as pd
from Bio import SeqIO
import mappy as mp
from tqdm import tqdm


def bases_at_ref_positions(hit, seq, target_rpos):
    """
    Return {ref_pos0: base_or_None} for target reference positions.

    Parameters
    ----------
    hit : mappy alignment hit
    seq : original full read sequence
    target_rpos : sorted iterable of 0-based reference positions

    Notes
    -----
    - Reference positions are always in forward-reference coordinates.
    - Returned bases are in reference orientation:
        * for + strand: the read base as-is
        * for - strand: the complemented read base, achieved by using the
          reverse-complemented query subsequence in alignment orientation
    - Deleted / uncovered target positions return None.
    """
    if not all(x <= y for x, y in zip(target_rpos, target_rpos[1:])):
        raise ValueError("target_rpos must be sorted in ascending order")

    out = []

    qseq = seq[hit.q_st : hit.q_en]
    if hit.strand == -1:
        qseq = mp.revcomp(qseq)

    qpos = 0
    rpos = hit.r_st
    ti = 0
    n_targets = len(target_rpos)

    # targets before aligned reference span
    while ti < n_targets and target_rpos[ti] < rpos:
        out.append(None)
        ti += 1

    for length, op in hit.cigar:
        if ti >= n_targets:
            break

        if op in (0, 7, 8):  # M, =, X
            block_r_end = rpos + length

            while ti < n_targets and target_rpos[ti] < block_r_end:
                p = target_rpos[ti]
                out.append(qseq[qpos + (p - rpos)])
                ti += 1

            qpos += length
            rpos = block_r_end

        elif op == 1:  # insertion in query
            qpos += length

        elif op in (2, 3):  # deletion / ref skip
            block_r_end = rpos + length

            while ti < n_targets and target_rpos[ti] < block_r_end:
                out.append(None)
                ti += 1

            rpos = block_r_end

        elif op == 4:  # soft clip
            qpos += length

        elif op in (5, 6):  # hard clip / pad
            pass

        else:
            raise ValueError(f"Unexpected CIGAR op: {op}")

    while ti < n_targets:
        out.append(None)
        ti += 1

    return out


def assign_reads_to_variants(
    alignments,
    read_seqs,
    alleles,
    SEQ_OFFSET=SEQ_OFFSET,
):
    """ """

    ref_nt_and_var_pos = list(
        sorted(set(var[:-2] for var in alleles), key=lambda x: int(x[:-1]))
    )
    expct_var_pos = [int(p[:-1]) for p in ref_nt_and_var_pos]
    expct_var_pos_0indx_and_offset = [(p - 1) + SEQ_OFFSET for p in expct_var_pos]
    ref_nt = np.array([p[-1] for p in ref_nt_and_var_pos])

    rows = []
    for seq_read_id, alignment in alignments.items():
        strand = "-" if alignment.strand == -1 else "+"
        # NOTE 0-based coordinates for the function
        rows.append(
            (
                seq_read_id,
                strand,
            )
            + tuple(
                bases_at_ref_positions(
                    alignment, read_seqs[seq_read_id], expct_var_pos_0indx_and_offset
                )
            )
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "seq_read_id",
            "strand",
        ]
        + ["nt_at_pos_" + str(p) for p in expct_var_pos],
    )

    df["assignment"] = pd.NA

    df.loc[
        (df.loc[:, df.columns.str.startswith("nt_at_pos_")] == ref_nt).all(axis=1),
        "assignment",
    ] = "REF"

    for var in alleles:
        alt_nt = var[-1]
        var_pos = var[:-3]
        # NOTE: this will overwrite in the case of multiple variants but
        # these reads are discarded in the lines below
        df.loc[df["nt_at_pos_" + var_pos] == alt_nt, "assignment"] = var
    n_vars_matched = reduce(
        lambda x, y: x + y,
        [(df["nt_at_pos_" + var[:-3]] == var[-1]).astype(int) for var in alleles],
    )
    if ((n_vars_matched >= 1) & (df["assignment"] == "REF")).any():
        raise UserWarning(
            "bug in assignment logic, read assigned as REF but has variant nt(s)"
        )

    df.loc[n_vars_matched > 1, "assignment"] = pd.NA

    # TODO: record rate of NA split by:
    # double mutants
    # REF with off-target mutations in target positions
    # REF with gaps in target positions

    return df


@lru_cache(maxsize=2048)
def load_aligner(ref_fasta_path):
    aligner = mp.Aligner(
        str(ref_fasta_path),
        preset="map-ont",
    )
    return aligner


def generate_alignments(input_fastq_path, ref_fasta_path):
    aligner = load_aligner(str(ref_fasta_path))
    if not aligner:
        raise RuntimeError("failed to load/build index")
    alignments = {}
    seqs = {}
    for name, seq, qual in mp.fastx_read(
        str(input_fastq_path)
    ):  # read a fasta/q sequence
        if name in seqs:
            raise ValueError(f"Duplicate read IDs: {name} in {input_fastq_path}")
        seqs[name] = seq
        for hit in aligner.map(seq):
            if hit.is_primary:
                alignments[name] = hit
                break
    return alignments, seqs


def filter_alignments(
    alignments, reference_size, orf_id_in_reference_file=None, MAX_MISMATCH=0.1
):
    alignments_filtered = {}
    for read_id, alignment in alignments.items():
        if (
            orf_id_in_reference_file is not None
            and alignment.ctg != orf_id_in_reference_file
        ):
            continue

        if alignment.mlen <= reference_size * (1 - MAX_MISMATCH):
            continue

        alignments_filtered[read_id] = alignment

    return alignments_filtered


@lru_cache(maxsize=2048)
def get_reference_size(ref_fasta_path):
    ref_seqs = [seq for seq in SeqIO.parse(ref_fasta_path, "fasta")]
    if len(ref_seqs) > 1:
        raise NotImplementedError(
            "more than one reference sequence in fasta not currently supported"
        )
    reference_size = len(ref_seqs[0].seq)
    return reference_size


def calculate_read_counts_for_one_well(
    input_fastq_path,
    alleles,
    ref_fasta_path,
    reference_size,
    SEQ_OFFSET=SEQ_OFFSET,
    return_all_stages=False,
):
    """
    TODO:
       - instead of pd.NA, have a few discarded_because_... including the reads etc.

    """
    sorted_index = ["REF", *sorted(alleles), pd.NA]

    if pd.isna(input_fastq_path):
        empty_counts = pd.Series(
            index=sorted_index, data=[0] * len(sorted_index), name="read_count"
        )
        empty_counts.index.name = "assignment"
        return empty_counts

    if reference_size is None:
        reference_size = get_reference_size(str(ref_fasta_path))

    alignments, seqs = generate_alignments(input_fastq_path, ref_fasta_path)
    alignments_filtered = filter_alignments(alignments, reference_size)
    # TODO: log the numbers filtered out
    df = assign_reads_to_variants(
        alignments_filtered, seqs, alleles, SEQ_OFFSET=SEQ_OFFSET
    )
    # TODO: log the number of discarded reads
    # TODO: log the +/- strand distribution of reads assigned to each variant
    counts = (
        df["assignment"]
        .value_counts(dropna=False)
        .rename("read_count")
        .reindex(sorted_index, fill_value=0)
    )

    if return_all_stages:
        return seqs, alignments, df, counts
    else:
        return counts


def load_allele_pools():
    """
    Return dict of {(experiment_id, orf_id, pool_id): set(nt_change)}.
    """
    alleles = {}
    df = pd.read_csv("../data/internal/pooled-Y2H_allele-pools.tsv", sep="\t")
    alleles = (
        df.groupby(["experiment", "orf_id", "pool_id"])["nt_change"]
        .apply(set)
        .to_dict()
    )
    return alleles


def load_plate_map():
    df = pd.read_csv("../data/internal/pooled-Y2H_plate-map.tsv", sep="\t")
    df["primerset"] = df["primerset_5"] + "__" + df["primerset_3"]

    if df[["experiment", "sample_name", "primerset", "well"]].duplicated().any():
        raise UserWarning("unexpected duplicates")
    if (
        df[["experiment", "orf_id", "pool_id", "interactor_id", "repeat_id", "media"]]
        .duplicated()
        .any()
    ):
        raise UserWarning("unexpected duplicates")

    return df


def add_file_locations_to_plate_map(
    plate_map,
    base_dir_well_binned_fastq="/data/bioinfo/rrm11/pooled_y2h_well_binned",
    base_dir_ref_fasta="../data/internal/CCSB_ORF_nt_seqs_12nt_surrounding",
):
    """
    TODO:
        - better name for args
    """
    # TODO: link this to "../data/internal/well_binned_fastq" or something
    base_dir_well_binned_fastq = Path(base_dir_well_binned_fastq)
    base_dir_ref_fasta = Path(base_dir_ref_fasta)
    if not base_dir_well_binned_fastq.exists():
        raise ValueError(f"{base_dir_well_binned_fastq} does not exist")
    if not base_dir_ref_fasta.exists():
        raise ValueError(f"{base_dir_ref_fasta} does not exist")

    def get_fastq_path(row):
        return (
            base_dir_well_binned_fastq
            / f"{row['experiment']}/{row['sample_name']}/{row['primerset']}/{row['well']}.subseq.fastq.gz"
        )

    def get_ref_fasta_path(orf_id):
        return base_dir_ref_fasta / f"{orf_id}_plus_12nt_each_end.fa"

    plate_map["fastq_path"] = plate_map.apply(get_fastq_path, axis=1)
    plate_map["ref_fasta_path"] = plate_map["orf_id"].apply(get_ref_fasta_path)

    if not plate_map["ref_fasta_path"].apply(lambda x: x.exists()).all():
        missing = plate_map.loc[
            ~plate_map["ref_fasta_path"].apply(lambda x: x.exists()), "ref_fasta_path"
        ].unique()
        raise UserWarning(
            f"Missing reference fasta files for the following ORFs: {missing}"
        )

    # TODO: log missing files
    # plate_map = plate_map[
    #    plate_map["fastq_path"].apply(
    #        lambda x: x.exists() and x.stat().st_size >= MIN_FASTQ_SIZE
    #    )
    # ].copy()
    plate_map.loc[
        ~plate_map["fastq_path"].apply(lambda x: x.exists()), "fastq_path"
    ] = pd.NA

    if plate_map["fastq_path"].isnull().all():
        raise UserWarning("No valid fastq files found for any wells")

    return plate_map


if __name__ == "__main__":
    start_time = time.perf_counter()
    CPU_TO_USE = 60

    alleles = load_allele_pools()
    plate_map = load_plate_map()
    plate_map = add_file_locations_to_plate_map(plate_map)

    experiments_with_files = set(
        plate_map.loc[plate_map["fastq_path"].notnull(), "experiment"].unique()
    )
    plate_map = plate_map.loc[plate_map["experiment"].isin(experiments_with_files), :]

    plate_map["alleles"] = plate_map.apply(
        lambda row: alleles[(row["experiment"], row["orf_id"], row["pool_id"])], axis=1
    )
    plate_map["ref_size"] = plate_map["ref_fasta_path"].apply(get_reference_size)

    njobs = CPU_TO_USE
    if len(plate_map) < CPU_TO_USE:
        njobs = len(plate_map)
    print(
        f"Using {njobs} parallel jobs to calculate read counts for {len(plate_map)} wells"
    )

    # hack to get the progress bar to work (otherwise would use starmap)
    def wrapper(args):
        return calculate_read_counts_for_one_well(*args)

    with Pool(njobs) as pool:
        results = list(
            tqdm(
                pool.imap(
                    wrapper,
                    plate_map[
                        ["fastq_path", "alleles", "ref_fasta_path", "ref_size"]
                    ].itertuples(index=False, name=None),
                ),
                total=len(plate_map),
            )
        )
    print("read count is done")
    key_cols = [
        "experiment",
        "orf_id",
        "pool_id",
        "interactor_id",
        "media",
        "repeat_id",
    ]
    keys = list(plate_map[key_cols].itertuples(index=False, name=None))
    df = (
        pd.concat(results, keys=keys)
        .reset_index()
        .rename(columns={f"level_{i}": col for i, col in enumerate(key_cols)})
    )
    # TODO: log these (above?)
    df = df.dropna(subset=["assignment"])

    df.to_csv(
        "../data/internal/pooled-Y2H_read-counts_all.tsv",
        sep="\t",
        index=False,
        na_rep="NULL",
    )
    # TODO: long-to-wide by media 3AT / -LW
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

# TODO
# - write results to database table
# - add a read_counts metadata table
#       - date,
#       - commit of the repo,
#       - and something about the input data
#           - hash?
#           - number of experiments, and experiment IDs?
