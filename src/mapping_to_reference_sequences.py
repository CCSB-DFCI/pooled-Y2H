import re
import time
import random
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from Bio import SeqIO, SwissProt, pairwise2
import requests
import tqdm
from joblib import Memory
from bioservices import BioMart

memory = Memory("../cache", verbose=0)


def generate_map_of_ensembl_to_hgnc_and_uniprot_ids(
    verbose=False,
    in_path=Path("../data/internal/CCSB_ORF_ID_to_GENCODE43_Ensembl_v109_mapping.tsv"),
    out_path=Path(
        "../data/internal/CCSB_ORF_ID_to_GENCODE43_Ensembl_v109_HGNC_UniProt_mapping.tsv"
    ),
):
    # this mapping is to GENCODE 43 / Ensembl 109
    # GRCh38.p13
    # freeze date: 08.2022
    # release date: 02.2023
    df = pd.read_csv(in_path, sep="\t")
    df["ensembl_protein_id"] = df["ensembl_protein_id"].str.split(".").str[0]
    if out_path.exists():
        prev = pd.read_csv(out_path, sep="\t")

        # remove from prev any orf ids with different ensg/ensp mappings
        n_rows_b4, n_cols_b4 = prev.shape
        prev = prev.merge(
            df[["orf_id", "ensembl_gene_id", "ensembl_protein_id"]],
            on="orf_id",
            how="left",
            suffixes=("", "_new"),
        )
        mask = prev["ensembl_gene_id_new"].isna() | (
            (prev["ensembl_gene_id"] == prev["ensembl_gene_id_new"])
            & (prev["ensembl_protein_id"] == prev["ensembl_protein_id_new"])
        )
        prev = prev.loc[mask, prev.columns[:n_cols_b4]]
        if verbose and (prev.shape[0] - n_rows_b4 > 0):
            print(
                f"{prev.shape[0] - n_rows_b4} previously mapped ORF IDs with updated Ensembl mappings."
            )

        df = df.loc[~df["orf_id"].isin(prev["orf_id"]), :]
        if df.shape[0] == 0:
            print("All ORFs already mapped. Doing nothing.")

    unmapped_ensg = df["ensembl_gene_id"].unique()
    unmapped_ensp = df["ensembl_protein_id"].unique()

    if verbose:
        print(df.shape[0], "ORF IDs to map.", prev.shape[0], "ORF IDs already mapped.")

    bm = BioMart(host="feb2023.archive.ensembl.org")
    bm.new_query()
    bm.add_dataset_to_xml("hsapiens_gene_ensembl")
    bm.add_filter_to_xml("ensembl_gene_id", ",".join(unmapped_ensg))
    bm.add_attribute_to_xml("ensembl_gene_id")
    bm.add_attribute_to_xml("hgnc_symbol")
    bm.add_attribute_to_xml("hgnc_id")
    tsv = bm.query(bm.get_xml())
    id_map = pd.read_csv(
        StringIO(tsv),
        sep="\t",
        header=None,
        names=[
            "ensembl_gene_id",
            "hgnc_symbol",
            "hgnc_id",
        ],
    )
    n_b4 = df.shape[0]
    df = pd.merge(df, id_map, on="ensembl_gene_id", how="left")
    if df.shape[0] != n_b4:
        print("WARNING: mapping issue")

    bm.new_query()
    bm.add_dataset_to_xml("hsapiens_gene_ensembl")
    bm.add_filter_to_xml("ensembl_peptide_id", ",".join(unmapped_ensp))
    bm.add_attribute_to_xml("ensembl_peptide_id")
    bm.add_attribute_to_xml("uniprotswissprot")
    tsv = bm.query(bm.get_xml())
    id_map = pd.read_csv(
        StringIO(tsv),
        sep="\t",
        header=None,
        names=[
            "ensembl_protein_id",
            "uniprot_ac_from_ensp",
        ],
    )
    n_b4 = df.shape[0]
    df = pd.merge(df, id_map, on="ensembl_protein_id", how="left")
    if df.shape[0] != n_b4:
        print("WARNING: mapping issue")

    bm.new_query()
    bm.add_dataset_to_xml("hsapiens_gene_ensembl")
    bm.add_filter_to_xml("ensembl_gene_id", ",".join(unmapped_ensg))
    bm.add_attribute_to_xml("ensembl_gene_id")
    bm.add_attribute_to_xml("uniprotswissprot")
    tsv = bm.query(bm.get_xml())
    id_map = pd.read_csv(
        StringIO(tsv),
        sep="\t",
        header=None,
        names=[
            "ensembl_gene_id",
            "uniprot_ac_from_ensg",
        ],
    )
    id_map = id_map.drop_duplicates().dropna(subset=["uniprot_ac_from_ensg"])
    # NOTE: some genes map to multiple uniprot acs
    df = pd.merge(df, id_map, on="ensembl_gene_id", how="left")

    gene_name_to_uniprot_ac = load_gene_name_to_uniprot_ac()

    df["uniprot_ac"] = (
        df["uniprot_ac_from_ensp"]
        .fillna(df["uniprot_ac_from_ensg"])
        .fillna(df["hgnc_symbol"].map(gene_name_to_uniprot_ac))
    )
    df = df.drop(
        columns=["uniprot_ac_from_ensp", "uniprot_ac_from_ensg"]
    ).drop_duplicates()
    df = pd.concat([df, prev], axis=0, ignore_index=True)
    if df["orf_id"].duplicated().any():
        raise UserWarning("unexpected duplicated orf_ids after ID mapping")
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        na_rep="NULL",
    )


@memory.cache
def load_gene_name_to_uniprot_ac():
    gene_name_to_uniprot_ac = {}
    for record in SeqIO.parse("../data/external/uniprot_sprot.fasta", format="fasta"):
        uniprot_ac = record.id.split("|")[1]
        match = re.search(r"GN=([^\s]+)", record.description)
        if match is None:
            continue
        gene_name = match.group(1)
        gene_name_to_uniprot_ac[gene_name] = uniprot_ac
    return gene_name_to_uniprot_ac


@memory.cache
def load_uniprot_canonical_isoforms():
    canonical_isoforms = {}
    with open("../data/external/uniprot_sprot.dat") as f:
        for record in SwissProt.parse(f):
            acc = record.accessions[0]
            comments = {c.split(":")[0]: c for c in record.comments}
            if "ALTERNATIVE PRODUCTS" in comments:
                match = re.search(
                    r"IsoId=([^;]+);\s+Sequence=Displayed;",
                    comments["ALTERNATIVE PRODUCTS"],
                )
                if match is None or len(match.groups()) == 0:
                    print(comments["ALTERNATIVE PRODUCTS"])
                    raise UserWarning(f"failed to extract isoform id for {acc}")
                elif len(match.groups()) >= 2:
                    print(comments["ALTERNATIVE PRODUCTS"])
                    raise UserWarning(
                        f"unexpected multiple canonical isoform ids for {acc}"
                    )
                canonical_isoforms[acc] = match.groups()[0].split(", ")[0]

    if not all(
        bool(re.fullmatch(rf"{re.escape(acc)}-\d{{1,3}}", iso))
        for acc, iso in canonical_isoforms.items()
    ):
        raise UserWarning("Some canonical isoform IDs do not match expected format.")

    return canonical_isoforms


def spdi_to_vep_variant(spdi: str) -> str:
    """
    Convert SPDI 'NC_0000xx.xx:pos:ref:alt' to VEP region format:
    'chrom pos . ref alt . . .'
    NOTE: assumes SNVs/indels where SPDI pos is 0-based interbase and VEP expects 1-based.
    """
    seq, pos0, ref, alt = spdi.split(":")
    pos1 = int(pos0) + 1

    if not seq.startswith("NC_0000"):
        raise NotImplementedError(
            f"Can't map contig {seq} (non-NC_0000..). Add a mapping table."
        )

    n = int(seq.split(".")[0].replace("NC_", ""))

    if 1 <= n <= 22:
        chrom = str(n)
    elif n == 23:
        chrom = "X"
    elif n == 24:
        chrom = "Y"
    else:
        raise NotImplementedError(f"Unhandled RefSeq contig {seq}")

    return f"{chrom} {pos1} . {ref} {alt} . . ."


@memory.cache
def post_vep_region(variants, params_items):
    max_retries = 6
    url = "https://feb2023.rest.ensembl.org/vep/homo_sapiens/region"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    for attempt in range(max_retries):
        r = requests.post(
            url,
            headers=headers,
            params=dict(params_items),
            json={"variants": variants},
            timeout=300,
        )

        # Success
        if r.status_code == 200:
            return r.json()

        # Rate limit / transient errors: backoff and retry
        if r.status_code in (429, 500, 502, 503, 504):
            sleep = (2**attempt) + random.random()
            time.sleep(sleep)
            continue

        # Hard failure: surface server's error JSON/text
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"VEP error {r.status_code}: {err}")

    raise RuntimeError("VEP repeatedly failed (rate limit / transient errors).")


def vep_mane_uniprot_from_spdi_batched(df, batch_size=200):
    """
    df must have columns: ccsb_allele_id, spdi
    Returns a DataFrame of transcript consequences, one row per transcript consequence.
    """
    params = {
        "mane": 1,
        "protein": 1,
        "uniprot": 1,
        "hgvs": 1,
        # no pick -> all transcripts
    }

    # Build variants + mapping back to allele_id
    allele_id_by_variant = {}
    variants = []
    for allele_id, spdi in zip(df["ccsb_allele_id"].values, df["spdi"].values):
        v = spdi_to_vep_variant(spdi)
        variants.append(v)
        allele_id_by_variant[v] = allele_id

    rows = []
    for i in tqdm.tqdm(range(0, len(variants), batch_size)):
        batch = variants[i : i + batch_size]
        # the tuples and sorting is to ensure consistent caching keys
        results = post_vep_region(
            tuple(batch), params_items=tuple(sorted(params.items()))
        )

        for rec in results:
            v_in = rec.get("input")  # matches submitted variant string
            allele_id = allele_id_by_variant.get(v_in)

            for tc in rec.get("transcript_consequences", []) or []:
                uniprot_iso = tc.get("uniprot_isoform", [None])[0]
                swissprot = tc.get("swissprot", [None])[0]
                uniprot_acc = (
                    uniprot_iso.split("-")[0]
                    if uniprot_iso
                    else (swissprot.split(".")[0] if swissprot else None)
                )

                rows.append(
                    {
                        "ccsb_allele_id": allele_id,
                        "vep_input": v_in,
                        "gene": tc.get("gene_symbol"),
                        "is_mane_select": "MANE_Select" in (tc.get("mane") or []),
                        "mane_refseq": tc.get("mane_select"),
                        "hgvsc": tc.get("hgvsc"),
                        "hgvsp": tc.get("hgvsp"),
                        "aa_change": tc.get("amino_acids"),
                        "protein_start": tc.get("protein_start"),
                        "protein_end": tc.get("protein_end"),
                        "uniprot_isoform": uniprot_iso,
                        "uniprot_accession": uniprot_acc,
                        "ensp": tc.get("protein_id"),
                        "enst": tc.get("transcript_id"),
                        "consequence_terms": ";".join(
                            tc.get("consequence_terms", []) or []
                        ),
                    }
                )

    return pd.DataFrame(rows)


def generate_transcript_consequences(out_path):
    # hiding import here to avoid circular imports
    from data_processing import load_variant_info

    vars = load_variant_info()
    tc = vep_mane_uniprot_from_spdi_batched(
        vars[["ccsb_allele_id", "spdi"]], batch_size=200
    )
    tc.to_csv(out_path, sep="\t", index=False)


def load_transcript_consequences():
    """
    TODO: as we add new variants, we don't want to rerun everything so
    we should add only the missing ones.
    """
    tc_path = Path(
        "../data/processed/transcript_consequences_from_VEP_API_Ensembl109.tsv"
    )
    if not tc_path.exists():
        print(
            "Calling Ensembl VEP API to generate transcript consequences. Will take several hours..."
        )
        generate_transcript_consequences(tc_path)
    tc = pd.read_csv(tc_path, sep="\t")
    tc["is_mane_select"] = tc["mane_refseq"].notnull()
    canonical_isoforms = load_uniprot_canonical_isoforms()
    tc["is_uniprot_canonical"] = tc["uniprot_isoform"].isin(
        set(canonical_isoforms.values())
    )
    tc["aa_change"] = (
        tc["hgvsp"].str.split(":").str[-1].str.replace("p.", "", regex=False)
    )
    return tc


def uniprot_style_diff_summary(algn):
    """
    Generate a UniProt-style difference summary.
    Canonical sequence is algn.seqB.
    """

    diffs = []

    pos_b = 0  # canonical position (1-based externally)
    i = 0
    n = len(algn.seqA)

    while i < n:
        a = algn.seqA[i]
        b = algn.seqB[i]

        if b != "-":
            pos_b += 1

        # --- Missing region
        if a == "-" and b != "-":
            start = pos_b
            seq = []
            while i < n and algn.seqA[i] == "-" and algn.seqB[i] != "-":
                seq.append(algn.seqB[i])
                i += 1
                pos_b += 1 if i < n and algn.seqB[i] != "-" else 0
            end = start + len(seq) - 1
            diffs.append(f"{start}-{end}: Missing")
            continue

        # --- Insertion
        if a != "-" and b == "-":
            anchor = pos_b
            ins = []
            while i < n and algn.seqA[i] != "-" and algn.seqB[i] == "-":
                ins.append(algn.seqA[i])
                i += 1
            ins_seq = "".join(ins)
            if anchor == 0:
                diffs.append(f"Inserted {ins_seq} (N-terminus)")
            else:
                diffs.append(f"{anchor}: Inserted {ins_seq}")
            continue

        # --- Substitution
        if a != "-" and b != "-" and a != b:
            start = pos_b
            canon = []
            iso = []

            while (
                i < n
                and algn.seqA[i] != "-"
                and algn.seqB[i] != "-"
                and algn.seqA[i] != algn.seqB[i]
            ):
                canon.append(algn.seqB[i])
                iso.append(algn.seqA[i])
                i += 1
                pos_b += 1 if i < n and algn.seqB[i] != "-" else 0

            end = start + len(canon) - 1
            diffs.append(f"{start}-{end}: {''.join(canon)} \u2192 {''.join(iso)}")
            continue

        i += 1

    return "\n".join(diffs)


def residue_mapping_from_aa_sequence_alignment(seq_a, seq_b, verbose=False):
    """
    NOTE: only maps if identical residues.
    """
    # The alignment parameters here shouldn't matter too much since
    # the mapping for a gap will be the same as the mapping for a mismatch
    # i.e. no mapping...
    alignment = pairwise2.align.globalms(
        seq_a, seq_b, 1, -0.1, -10, -0.5, penalize_end_gaps=False
    )
    if verbose:
        print(pairwise2.format_alignment(*alignment[0]))
    if verbose:
        print(
            "Differences from UniProt canonical isoform:\n"
            + uniprot_style_diff_summary(alignment[0])
        )

    mapping = {}
    pos_a = 0
    pos_b = 0
    for a, b in zip(alignment[0].seqA, alignment[0].seqB):
        if a != "-":
            pos_a += 1
        if b != "-":
            pos_b += 1

        if a == "-":
            continue
        else:
            if a == b:
                mapping[pos_a] = pos_b
            elif b == "-":
                mapping[pos_a] = None
            elif a != b:
                mapping[pos_a] = None
            else:
                raise UserWarning("unexpected case")

    return mapping


def map_cloned_orf_to_uniprot_aa_change(
    orf_id,
    uniprot_iso_ac,
    aa_changes,
    allele_ids,
    orf_seq,
    uniprot_seq,
    tc=None,
    verbose=False,
):
    """
    # TODO: check format of aa_changes argument
    """
    if len(aa_changes) != len(allele_ids):
        raise ValueError("aa_changes and allele_ids must have the same length")
    if uniprot_iso_ac is None or pd.isnull(uniprot_iso_ac):
        if verbose:
            print(f"no uniprot_ac for {orf_id}")
        return [None] * len(aa_changes)
    # perfect match to uniprot canonical
    if orf_seq == uniprot_seq:
        if verbose:
            print("perfect match")
        return aa_changes
    # same length, just check for differences
    elif len(orf_seq) == len(uniprot_seq):
        if verbose:
            print("same length with at least some differences")
        aa_changes_uniprot = []
        for aa_change in aa_changes:
            pos = int(aa_change[3:-3])
            if orf_seq[pos - 1] == uniprot_seq[pos - 1]:
                aa_changes_uniprot.append(aa_change)
            else:
                aa_changes_uniprot.append(None)
        return aa_changes_uniprot

    tc_rows = (tc["uniprot_isoform"] == uniprot_iso_ac) & (
        tc["consequence_terms"].str.contains("missense_variant")
    )
    # if available use genomic mapping from ensembl VEP
    if tc_rows.any():
        if verbose:
            print(f"mapping via ensembl VEP for {orf_id} {uniprot_iso_ac}")
        aa_changes_uniprot = []
        for aa_change_orf, allele_id in zip(aa_changes, allele_ids):
            matched_changes = tc.loc[
                tc_rows & (tc["ccsb_allele_id"] == allele_id), "aa_change"
            ].unique()
            if len(matched_changes) == 0:
                aa_changes_uniprot.append(None)
            elif len(matched_changes) == 1:
                aa_changes_uniprot.append(matched_changes[0])
            else:
                # in rare cases multiple ensembl transcripts can map to the same
                # uniprot AC in different positions
                if aa_change_orf in matched_changes:
                    aa_changes_uniprot.append(aa_change_orf)
                else:
                    # here I'm not sure what to do
                    raise UserWarning(
                        f"multiple matched a.a. changes for {orf_id} {uniprot_iso_ac} {aa_change_orf}: {matched_changes}"
                    )
        return aa_changes_uniprot
    # Last resort: align a.a. sequences (some uniprot canonical don't have genomic mapping)
    else:
        orf_to_uniprot_map = residue_mapping_from_aa_sequence_alignment(
            orf_seq, uniprot_seq, verbose=verbose
        )
        aa_changes_uniprot = []
        for aa_change_orf in aa_changes:
            pos_orf = int(aa_change_orf[3:-3])
            if (
                pos_orf in orf_to_uniprot_map
                and orf_to_uniprot_map[pos_orf] is not None
            ):
                aa_changes_uniprot.append(
                    f"{aa_change_orf[:3]}{orf_to_uniprot_map[pos_orf]}{aa_change_orf[-3:]}"
                )
            else:
                aa_changes_uniprot.append(None)
        return aa_changes_uniprot


def map_to_uniprot_canonical_aa_positions(
    group,
    clone_seqs=None,
    uniprot_seqs=None,
    tc=None,
    canonical_isoforms=None,
    verbose=False,
):
    if verbose:
        print(group.name)
    orf_id = group.name
    uniprot_ac = group["uniprot_ac"].values[0]
    uniprot_iso_ac = canonical_isoforms.get(uniprot_ac, uniprot_ac)
    aa_changes = group["aa_change_cloned_orf"].values.tolist()
    allele_ids = group["ccsb_allele_id"].values.tolist()
    aa_changes_uniprot = map_cloned_orf_to_uniprot_aa_change(
        orf_id,
        uniprot_iso_ac,
        aa_changes,
        allele_ids,
        orf_seq=clone_seqs.loc[clone_seqs["orf_id"] == orf_id, "p_seq"].values[0],
        uniprot_seq=uniprot_seqs.get(uniprot_ac, None),
        tc=tc,
        verbose=verbose,
    )
    return pd.Series(aa_changes_uniprot, index=group.index)


def map_to_uniprot_isoform_aa_positions(
    group, clone_seqs=None, uniprot_seqs=None, tc=None, verbose=False
):
    if verbose:
        print(group.name)

    if group["wt_orf_matches_uniprot_canonical"].all():
        # already mapped
        return group["aa_change_uniprot_canonical"]

    orf_id = group.name
    uniprot_iso_ac = group["uniprot_isoform_ac"].values[0]
    aa_changes = group["aa_change_cloned_orf"].values.tolist()
    allele_ids = group["ccsb_allele_id"].values.tolist()
    aa_changes_uniprot = map_cloned_orf_to_uniprot_aa_change(
        orf_id,
        uniprot_iso_ac,
        aa_changes,
        allele_ids,
        orf_seq=clone_seqs.loc[clone_seqs["orf_id"] == orf_id, "p_seq"].values[0],
        uniprot_seq=uniprot_seqs.get(uniprot_iso_ac, None),
        tc=tc,
        verbose=verbose,
    )
    return pd.Series(aa_changes_uniprot, index=group.index)


def align_seqs(seq_a, seq_b):
    algn = pairwise2.align.globalms(
        seq_a, seq_b, 1, -0.1, -10, -0.5, penalize_end_gaps=False
    )[0]
    return algn


def simple_score(algn):

    def score(a, b):
        if a == b:
            return 1
        else:
            return -1

    return sum(score(a, b) for a, b in zip(algn.seqA, algn.seqB))


def count_mismatches(algn):
    return sum(a != b for a, b in zip(algn.seqA, algn.seqB))


def uniprot_match_details(
    orf_id,
    uniprot_ac,
    clone_seqs=None,
    uniprot_canonical_seqs=None,
    uniprot_alt_seqs=None,
    canonical_isoforms=None,
):
    """
    return matches canonical bool, alt ac string, match details string

    """
    if uniprot_ac is None or pd.isnull(uniprot_ac):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    aa_seq_clone = clone_seqs.loc[clone_seqs["orf_id"] == orf_id, "p_seq"].values[0]
    aa_seq_uniprot_canonical = uniprot_canonical_seqs[uniprot_ac]
    if aa_seq_clone == aa_seq_uniprot_canonical:
        return (
            True,
            0,
            100,
            canonical_isoforms.get(uniprot_ac, uniprot_ac),
            f"Perfect match to UniProt canonical sequence {uniprot_ac}.",
        )

    aa_seqs_alt = {
        acc: seq
        for acc, seq in uniprot_alt_seqs.items()
        if acc.startswith(uniprot_ac + "-") and seq != aa_seq_uniprot_canonical
    }
    alignments = {uniprot_ac: align_seqs(aa_seq_clone, aa_seq_uniprot_canonical)}
    for alt_acc, alt_seq in aa_seqs_alt.items():
        alignments[alt_acc] = align_seqs(aa_seq_clone, alt_seq)

    if len(aa_seqs_alt) == 0:
        algn = alignments[uniprot_ac]
        n_mismatches = count_mismatches(algn)
        pct_id = (1 - n_mismatches / len(algn.seqA)) * 100
        desc = uniprot_style_diff_summary(algn)
        return (
            True,
            n_mismatches,
            pct_id,
            canonical_isoforms.get(uniprot_ac, uniprot_ac),
            f"Matched to UniProt canonical isoform.\nDifferences:\n{desc}",
        )

    for alt_acc, alt_seq in aa_seqs_alt.items():
        if aa_seq_clone == alt_seq:
            return (
                False,
                0,
                100,
                alt_acc,
                f"Perfect match to UniProt alternative isoform {alt_acc}.",
            )

    best_score = max(simple_score(algn) for algn in alignments.values())
    if sum(simple_score(algn) == best_score for algn in alignments.values()) > 1:
        raise UserWarning(
            f"Multiple equally good matches to UniProt isoforms ORF ID: {orf_id}, {[acc for acc, algn in alignments.items() if simple_score(algn) == best_score]}."
        )
    best_acc = [
        acc for acc, algn in alignments.items() if simple_score(algn) == best_score
    ][0]
    if best_acc == uniprot_ac:
        algn = alignments[uniprot_ac]
        n_mismatches = count_mismatches(algn)
        pct_id = (1 - n_mismatches / len(algn.seqA)) * 100
        desc = uniprot_style_diff_summary(algn)
        return (
            True,
            n_mismatches,
            pct_id,
            canonical_isoforms.get(uniprot_ac, uniprot_ac),
            f"Matched to UniProt canonical isoform {uniprot_ac}.\nDifferences:\n{uniprot_style_diff_summary(alignments[uniprot_ac])}",
        )
    else:
        algn = alignments[best_acc]
        n_mismatches = count_mismatches(algn)
        pct_id = (1 - n_mismatches / len(algn.seqA)) * 100
        desc = uniprot_style_diff_summary(algn)
        return (
            False,
            n_mismatches,
            pct_id,
            best_acc,
            f"Matched to UniProt alternative isoform {best_acc}. Differences:\n{uniprot_style_diff_summary(alignments[best_acc])}",
        )


def is_standard_aa_seq(seq):
    standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(r in standard_aas for r in seq):
        msg = f"Non-standard amino acid found in sequence: {seq}\n"
        msg += "\n".join(
            [f"{r} at pos {i + 1}" for i, r in enumerate(seq) if r not in standard_aas]
        )
        raise ValueError(msg)


@memory.cache
def add_uniprot_aa_pos_mapping(df):
    df = df.copy()
    aa_seqs = pd.read_csv("../data/internal/CCSB_ORF_aa_seqs.tsv", sep="\t")
    if aa_seqs["orf_id"].isnull().any():
        raise UserWarning("unexpected missing values")
    if aa_seqs["orf_id"].duplicated().any():
        raise UserWarning("unexpected duplicated orf_ids")
    uniprot_seqs = {
        record.id.split("|")[1]: str(record.seq)
        for record in SeqIO.parse(
            "../data/external/uniprot_sprot.fasta", format="fasta"
        )
    }
    canonical_isoforms = load_uniprot_canonical_isoforms()
    uniprot_alt_seqs = {
        record.id.split("|")[1]: str(record.seq)
        for record in SeqIO.parse(
            "../data/external/uniprot_sprot_varsplic.fasta", format="fasta"
        )
    }

    df["aa_change_cloned_orf"] = df["aa_change"]
    # this ID map is for both disease genes and interactors
    # but is missing some from the variant collection
    id_map = pd.read_csv(
        "../data/internal/CCSB_ORF_ID_to_GENCODE43_Ensembl_v109_HGNC_UniProt_mapping.tsv",
        sep="\t",
    )
    if id_map["orf_id"].duplicated().any():
        raise UserWarning("unexpected duplicate ORF IDs in mapping table")

    n_b4 = df.shape[0]
    # Adding ensembl, uniprot and HGNC IDs
    df = pd.merge(df, id_map, on="orf_id", how="left")
    if df.shape[0] != n_b4:
        raise UserWarning("unexpected change in number of variants after merge")

    # This fails at the moment because some variants in the table are not
    # in Tong's ORF mapping table
    # if df['uniprot_ac'].isnull().any():
    #    raise UserWarning("unexpected missing uniprot accession after merge")

    tc = load_transcript_consequences()
    # TODO: check tc has all variants? I think it won't though...
    # in that case save the variants tested somewhere...

    uniprot_matches = pd.DataFrame(
        data=[
            (
                orf_id,
                *uniprot_match_details(
                    orf_id,
                    uniprot_ac,
                    clone_seqs=aa_seqs,
                    uniprot_canonical_seqs=uniprot_seqs,
                    uniprot_alt_seqs=uniprot_alt_seqs,
                    canonical_isoforms=canonical_isoforms,
                ),
            )
            for orf_id, uniprot_ac in df.loc[:, ["orf_id", "uniprot_ac"]]
            .drop_duplicates()
            .values
        ],
        columns=[
            "orf_id",
            "wt_orf_matches_uniprot_canonical",
            "n_mismatches_with_uniprot_sequence",
            "pct_identity_with_uniprot_sequence",
            "uniprot_isoform_ac",
            "wt_orf_uniprot_sequence_match_details",
        ],
    )
    if uniprot_matches["orf_id"].duplicated().any():
        raise UserWarning("unexpected duplicate orf_id in uniprot_matches")
    df = pd.merge(df, uniprot_matches, on="orf_id", how="left")

    out = df.groupby("orf_id", group_keys=False)[
        ["uniprot_ac", "aa_change_cloned_orf", "ccsb_allele_id"]
    ].apply(
        map_to_uniprot_canonical_aa_positions,
        clone_seqs=aa_seqs,
        uniprot_seqs=uniprot_seqs,
        tc=tc,
        canonical_isoforms=canonical_isoforms,
    )
    df["aa_change_uniprot_canonical"] = out

    out = df.groupby("orf_id", group_keys=False)[
        [
            "uniprot_isoform_ac",
            "wt_orf_matches_uniprot_canonical",
            "aa_change_cloned_orf",
            "aa_change_uniprot_canonical",
            "ccsb_allele_id",
        ]
    ].apply(
        map_to_uniprot_isoform_aa_positions,
        clone_seqs=aa_seqs,
        uniprot_seqs=uniprot_alt_seqs,
        tc=tc,
    )
    df["aa_change_uniprot_isoform"] = out

    return df


def test_uniprot_mappings(df):
    canonical_100pct_match = df["wt_orf_matches_uniprot_canonical"] & (
        df["pct_identity_with_uniprot_sequence"] == 100
    )
    if not (
        df.loc[canonical_100pct_match, "aa_change_cloned_orf"]
        == df.loc[canonical_100pct_match, "aa_change_uniprot_canonical"]
    ).all():
        raise UserWarning("Inconsistency with uniprot canonical mappings.")
    if not (
        df.loc[canonical_100pct_match, "aa_change_uniprot_isoform"]
        == df.loc[canonical_100pct_match, "aa_change_uniprot_canonical"]
    ).all():
        raise UserWarning("Inconsistency with uniprot canonical mappings.")
    if not (
        (df["uniprot_isoform_ac"].str.split("-").str[0] == df["uniprot_ac"])
        | df["uniprot_ac"].isnull()
    ).all():
        raise UserWarning("Inconsistency between uniprot_ac and uniprot_isoform_ac")
    if not (df["uniprot_isoform_ac"].isnull() == df["uniprot_ac"].isnull()).all():
        raise UserWarning("Inconsistency between uniprot_ac and uniprot_isoform_ac")
    if (df["pct_identity_with_uniprot_sequence"] > 100).any():
        raise UserWarning("pct_identity_with_uniprot_sequence > 100 found")
    if (df["pct_identity_with_uniprot_sequence"] < 0).any():
        raise UserWarning("pct_identity_with_uniprot_sequence < 0 found")
    canonical_match = (df["wt_orf_matches_uniprot_canonical"] == True) & df[
        "aa_change_uniprot_canonical"
    ].notnull()
    if not (
        df.loc[canonical_match, "aa_change_uniprot_isoform"]
        == df.loc[canonical_match, "aa_change_uniprot_canonical"]
    ).all():
        raise UserWarning("Inconsistency with uniprot isoform mappings")
