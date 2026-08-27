import functools

import networkx as nx
import numpy as np
import obonet
import pandas as pd
from joblib import Memory

memory = Memory("../cache", verbose=0)


@memory.cache
def add_clinvar_phenotypes(df):
    """"""
    df = df.copy()
    cv = load_clinvar_snps()

    n_b4 = df.shape[0]
    df = pd.merge(
        df,
        cv.loc[
            :,
            [
                "spdi",
                "#AlleleID",
                "ClinicalSignificance",
                "PhenotypeIDS",
                "PhenotypeList",
                "Origin",
                "OriginSimple",
                "ReviewStatus",
            ],
        ],
        how="left",
        on="spdi",
    )
    if df.shape[0] != n_b4:
        raise UserWarning("Merge changed number of rows")

    df["cv_allele_id"] = df["#AlleleID"]
    df["cv_allele_id"] = df["cv_allele_id"].astype(pd.Int64Dtype())
    df["cv_clinical_significance"] = df["ClinicalSignificance"]
    df["cv_phenotypes"] = df["PhenotypeList"]
    df["cv_phenotype_ids"] = df["PhenotypeIDS"]
    df["cv_origin"] = df["Origin"]
    df["cv_origin_simple"] = df["OriginSimple"]
    df["cv_review_status"] = df["ReviewStatus"]
    df["cv_clinical_significance_clean"] = df["ClinicalSignificance"].map(
        lambda x: {
            "Pathogenic": "Pathogenic",
            "Likely pathogenic": "Likely pathogenic",
            "Pathogenic/Likely pathogenic": "Likely pathogenic",
            "Benign": "Benign",
            "Likely benign": "Likely benign",
            "Benign/Likely benign": "Likely benign",
            "Uncertain significance": "VUS",
            "Conflicting classifications of pathogenicity": "Conflicting",
        }.get(x, "Other")
    )
    df["clinical_significance_simple"] = (
        df["cv_clinical_significance_clean"]
        .map(
            {
                "Pathogenic": "Pathogenic",
                "Likely pathogenic": "Pathogenic",
                "Benign": "Benign",
                "Likely benign": "Benign",
                "Conflicting": "Conflicting",
                "VUS": "VUS",
                "Other": "Other",
            }
        )
        .fillna("Other")
    )

    return df


@memory.cache
def load_clinvar_snps():
    cv = pd.read_csv(
        "../data/external/clinvar_variant_summary_2025-08-10.txt",
        low_memory=False,
        sep="\t",
    )
    cv["#AlleleID"] = cv["#AlleleID"].astype(pd.Int64Dtype())
    cv = cv.loc[
        (cv["Assembly"] == "GRCh38") & (cv["Type"] == "single nucleotide variant"), :
    ]
    # NOTE: this only works for SNPs
    cv["spdi"] = (
        cv["ChromosomeAccession"]
        + ":"
        + (cv["PositionVCF"] - 1).astype(str)
        + ":"
        + cv["ReferenceAlleleVCF"]
        + ":"
        + cv["AlternateAlleleVCF"]
    )
    return cv


@memory.cache
def load_mondo_ontology():

    def filter_mondo_graph(G):
        H = nx.DiGraph()

        for n, data in G.nodes(data=True):
            if not isinstance(n, str) or not n.startswith("MONDO:"):
                continue

            H.add_node(n, **data)

        for u, v, k in G.edges(keys=True):
            if k != "is_a":
                continue
            if u in H and v in H:
                H.add_edge(u, v, key="is_a")

        return H

    mondo_raw = obonet.read_obo("../data/external/mondo.obo")
    mondo = filter_mondo_graph(mondo_raw)
    return mondo


@memory.cache
def load_mondo_id_mapping():
    mapping_to_mondo = {}
    mondo_raw = obonet.read_obo("../data/external/mondo.obo")
    for mondo_id, node_data in mondo_raw.nodes(data=True):
        if not mondo_id.startswith("MONDO:"):
            continue

        # include self-mapping
        mapping_to_mondo[mondo_id] = set([mondo_id])

        if "xref" not in node_data:
            continue
        for xref in node_data["xref"]:
            mapping_to_mondo.setdefault(xref, set()).add(mondo_id)
    return mapping_to_mondo


@memory.cache
def add_mondo_ids(df):
    """
    TODO:
        - add HGMD to MONDO. They don't have a mapping because
          HGMD is proprietary.

    """
    df = df.copy()
    mondo = load_mondo_ontology()
    mapping_to_mondo = load_mondo_id_mapping()

    df["mondo_ids"] = (
        df["cv_phenotype_ids"]
        .str.split("[;,|]", regex=True)
        .apply(map_clinvar_to_mondo, args=(mapping_to_mondo,))
    )
    df["n_mondo_ids"] = (df["mondo_ids"].str.count(";") + 1).fillna(0).astype(int)

    df["filtered_mondo_ids"] = df.groupby("symbol")["mondo_ids"].transform(
        consolidate_mondo_terms_for_a_gene, mondo=mondo
    )
    df["filtered_mondo_phenotypes"] = df["filtered_mondo_ids"].apply(
        lambda x: "|".join(
            [mondo.nodes[mid]["name"] for mid in x.split(";")] if pd.notna(x) else []
        )
    )
    if (df["mondo_ids"].isnull() & df["filtered_mondo_ids"].notnull()).any():
        raise UserWarning(
            "Something wrong: some variants have filtered MONDO IDs but no input MONDO IDs"
        )
    if (df["mondo_ids"].notnull() & df["filtered_mondo_ids"].isnull()).any():
        raise UserWarning(
            "Something wrong: some variants have input MONDO IDs but no filtered MONDO IDs"
        )
    df["n_filtered_mondo_ids"] = (
        (df["filtered_mondo_ids"].str.count(";") + 1).fillna(0).astype(int)
    )

    df["high_level_mondo_terms"] = df["filtered_mondo_ids"].apply(
        high_level_mondo_terms, mondo=mondo
    )

    return df


def map_clinvar_to_mondo(phenotype_ids, mapping_to_mondo, debug=False):
    if phenotype_ids is np.nan:
        return np.nan
    if len(phenotype_ids) == 0:
        return np.nan
    # Format ClinVar phenotype IDs to match those in MONDO xrefs
    # For some reason they're not consistent so have to do a bunch
    # of crap to get them to match.
    formated_ids = []
    for pid in phenotype_ids:
        if pid.startswith("MedGen:C"):
            formated_ids.append(pid.replace("MedGen:C", "UMLS:C"))
        elif pid.startswith("MedGen:"):
            formated_ids.append(pid.replace("MedGen:", "MEDGEN:"))
        elif pid.startswith("Orphanet:ORPHA"):
            formated_ids.append(pid.replace("Orphanet:ORPHA", "Orphanet:"))
        elif pid.startswith("SNOMED CT:"):
            formated_ids.append(pid.replace("SNOMED CT:", "SCTID:"))
        elif pid.startswith("Human Phenotype Ontology:HP:"):
            formated_ids.append(pid.replace("Human Phenotype Ontology:HP:", "HP:"))
        elif pid.startswith("MeSH:"):
            formated_ids.append(pid.replace("MeSH:", "MESH:"))
        elif pid.startswith("MONDO:MONDO:"):
            formated_ids.append(pid.replace("MONDO:MONDO:", "MONDO:"))
        else:
            formated_ids.append(pid)

    if debug:
        uninformative_ids = {
            "UMLS:C3661900",  # 'not provided'
            "UMLS:CN517202",  # 'not provided'?
            "UMLS:CN169374",  # 'not specified'
            "UMLS:C0950123",  # inborn genetic diseases
            "na",
            "-",
            "",
        }
        unmapped = [
            pid
            for pid in formated_ids
            if pid not in mapping_to_mondo and pid not in uninformative_ids
        ]
        if len(unmapped) > 0:
            print("IDs that failed mapping:", unmapped)

    mondo_ids = set.union(*[mapping_to_mondo.get(pid, set()) for pid in formated_ids])
    if len(mondo_ids) == 0:
        return np.nan
    return ";".join(sorted(mondo_ids))


def high_level_mondo_terms(mondo_ids, mondo=None):
    """
    NOTE: not every term is caught by this method. E.g. it misses terms that
    are purely sucesptability to diseases, or terms that are descendants of only
    the 'hereditary disease' term. But those are relatively few.
    """
    if pd.isna(mondo_ids):
        return np.nan
    mondo_ids = mondo_ids.split(";")
    if mondo is None:
        mondo = obonet.read_obo("../data/external/mondo.obo")

    HUMAN_DISEASE = "MONDO:0700096"
    # NOTE: predecessors/ancestors is the oppostite direction you would guess
    high_level_terms = set(mondo.predecessors(HUMAN_DISEASE))
    terms_to_remove = {"MONDO:0003847"}  # hereditary disease
    high_level_terms = high_level_terms.difference(terms_to_remove)

    result = set()
    for term in mondo_ids:
        upstream_terms = nx.descendants(mondo, term)
        upstream_terms.add(term)
        result.update(upstream_terms.intersection(high_level_terms))
    if len(result) == 0:
        return np.nan
    names = [mondo.nodes[mid]["name"] for mid in result]
    return ";".join(sorted(names))


def prune_redundant_terms(mondo, terms):
    """
    Remove any term that is an ancestor of another term in the set.
    Keeps the most specific set.
    """
    terms = [t for t in terms if t in mondo]
    term_set = set(terms)

    redundant = set()
    for t in term_set:
        # if any other term is a descendant (more specific) of t, then t is redundant
        # descendants in MONDO = nx.ancestors because edges child->parent
        more_specific = nx.ancestors(mondo, t)
        if (more_specific & term_set) - {t}:
            redundant.add(t)

    return sorted(term_set - redundant)


def most_specific_common_superterm(mondo, terms):
    """Deepest common ancestor (more general term) among all `terms`."""
    if not terms:
        return None

    common = None
    for t in terms:
        st = set(nx.descendants(mondo, t))  # follow term -> parents -> ...
        st.add(t)
        common = st if common is None else common.intersection(st)
        if not common:
            return None

    return min(common, key=lambda x: n_subterms(x, mondo))


@functools.cache
def n_subterms(mondo_id, mondo):
    return len(nx.ancestors(mondo, mondo_id))


def collapse_terms(
    terms, mondo, target_min_n_terms=1, max_n_subterms=30, verbose=False
):
    """
    Collapse a set of MONDO terms to <= k by merging pairs upward.
    Won't merge into anything shallower than min_depth (too general).
    """
    terms_out = list(terms)

    if len(terms_out) <= target_min_n_terms:
        return terms_out

    # greedy pairwise merging
    while len(terms_out) > target_min_n_terms:
        best = None  # (gain, merge_depth, i, j, msca)
        for i in range(len(terms_out)):
            for j in range(i + 1, len(terms_out)):
                a, b = terms_out[i], terms_out[j]
                msca = most_specific_common_superterm(mondo, [a, b])
                if msca is None:
                    continue

                d = n_subterms(msca, mondo)
                if d > max_n_subterms:
                    continue  # would be too general

                # "gain" is always 1 term reduced; tie-break by choosing more specific merge (larger depth)
                candidate = (1, d, i, j, msca)
                if best is None or candidate[1] < best[1]:
                    best = candidate

        if best is None:
            break  # no acceptable merges without going too general

        _, _, i, j, msca = best
        if verbose:
            print(
                f"merging {terms_out[i]} ({mondo.nodes[terms_out[i]]['name']}) and {terms_out[j]} ({mondo.nodes[terms_out[j]]['name']}) into {msca} ({mondo.nodes[msca]['name']})"
            )
        # replace the pair with the merged term
        new = []
        for idx, t in enumerate(terms_out):
            if idx not in (i, j):
                new.append(t)
        new.append(msca)
        terms_out = new

    return sorted(terms_out)


def consolidate_mondo_terms_for_a_gene(
    input_mondo_ids_column_for_gene,
    mondo,
    target_min_n_terms=1,
    max_n_subterms=30,
    verbose=False,
):
    variant_mondo_ids = input_mondo_ids_column_for_gene.str.split(";").to_list()
    variant_mondo_ids = [ts if isinstance(ts, list) else [] for ts in variant_mondo_ids]
    input_terms = set([t for ts in variant_mondo_ids for t in ts])
    if verbose:
        print(f"input, {len(input_terms)} terms:")
        print([(t, mondo.nodes[t]["name"]) for t in sorted(input_terms)])

    pruned_variant_mondo_ids = [
        prune_redundant_terms(mondo, ts) if len(ts) > 0 else []
        for ts in variant_mondo_ids
    ]
    pruned_terms = set([t for ts in pruned_variant_mondo_ids for t in ts])
    if verbose:
        print(f"after per-variant pruning, {len(pruned_terms)} terms:")
        print([(t, mondo.nodes[t]["name"]) for t in sorted(pruned_terms)])

    filtered_terms = collapse_terms(
        pruned_terms,
        mondo,
        target_min_n_terms=target_min_n_terms,
        max_n_subterms=max_n_subterms,
        verbose=verbose,
    )
    if verbose:
        print(f"output, {len(filtered_terms)} terms:")
        print([(t, mondo.nodes[t]["name"]) for t in sorted(filtered_terms)])

    term_map = {}
    for t in pruned_terms:
        if t in filtered_terms:
            term_map[t] = t
        else:
            # find most specific related term
            related = [
                ft
                for ft in filtered_terms
                if nx.has_path(mondo, t, ft) or nx.has_path(mondo, ft, t)
            ]
            if len(related) == 0:
                raise UserWarning(
                    f"Input term {t} maps to none of output terms {filtered_terms}"
                )
            term_map[t] = min(related, key=lambda x: n_subterms(x, mondo))

    output = pd.Series(
        index=input_mondo_ids_column_for_gene.index,
        data=[
            ";".join(sorted(set([term_map[t] for t in ts]))) if ts else np.nan
            for ts in pruned_variant_mondo_ids
        ],
    )

    return output


def add_mode_of_inheritance(df):
    df = df.copy()
    gencc = pd.read_csv("../data/external/gencc-submissions_2025-10-30.tsv", sep="\t")

    tmp = (
        df.reset_index(drop=False)
        .rename(columns={"index": "row_id"})[["row_id", "symbol", "mondo_ids"]]
        .copy()
    )
    tmp["mondo_list"] = (
        tmp["mondo_ids"]
        .where(pd.notna(tmp["mondo_ids"]), "")
        .astype(str)
        .str.split(";")
    )
    tmp = tmp.explode("mondo_list").rename(columns={"mondo_list": "mondo_id"})
    tmp = pd.merge(
        tmp,
        gencc,
        left_on=["symbol", "mondo_id"],
        right_on=["gene_symbol", "disease_curie"],
        how="left",
    )

    def summarize_moi(series):
        mois = sorted(set(series.dropna()))
        if len(mois) == 0:
            return np.nan
        if len(mois) == 1:
            return mois[0]
        else:
            return "conflicting: " + ";".join(mois)

    tmp = tmp.groupby("row_id")["moi_title"].apply(summarize_moi).rename("moi")

    df = pd.merge(df, tmp, left_index=True, right_index=True, how="left")
    return df
