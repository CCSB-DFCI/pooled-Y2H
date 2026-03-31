import random
import time
from pathlib import Path

import requests
import tqdm
import pandas as pd

API_URL = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r4"
# ---- Tuning knobs ----
BATCH_SIZE = 20  # it complained when I tried 50
BASE_SLEEP = 1.0
MAX_RETRIES = 6  # per request
BACKOFF_FACTOR = 2.0  # exponential backoff multiplier (1, 2, 4, 8, 16...)
JITTER = 0.5  # random jitter added to sleep to avoid being "too regular"
TIMEOUT_SEC = 120  # it timed out at 30s
# ----------------------


def add_allele_frequency(df):
    """
    I'm just using the exome allele frequency for now.
    My reasoning is: the exome dataset is larger than the geonome
    and adding the genome only adds a small number of variants.

    There are some with allele frequency of 0, which I don't understand.
    I could also put a floor of something like 1 / number of sequenced exomes
    for the missing data...

    """
    df = df.copy()
    var_ids = df.apply(build_variant_id, axis=1)
    gnomad = load_gnomad_allele_counts(var_ids.to_list())
    df["allele_frequency"] = var_ids.map(gnomad["af_exome"])
    return df


def load_gnomad_allele_counts(var_ids):
    cache_file = Path("../data/external/gnomad_v4_allele_counts_from_API.tsv")
    if cache_file.exists():
        df = pd.read_csv(cache_file, sep="\t", index_col="variant_id")
    else:
        df = pd.DataFrame()
    vars_with_data = set(df.index)
    if set(var_ids).issubset(vars_with_data):
        return df
    else:
        missing_vars = set(var_ids).difference(vars_with_data)
        df = pd.concat([df, _query_gnomad_allele_counts(missing_vars)])
        df.sort_index().to_csv(
            cache_file, sep="\t", index=True, index_label="variant_id"
        )
    return df


def _query_gnomad_allele_counts(var_ids):
    results = {}
    print("Querying gnomAD API for allele frequencies...")
    for i in tqdm.tqdm(range(0, len(var_ids), BATCH_SIZE)):
        batch = var_ids[i : i + BATCH_SIZE]
        res = query_gnomad(batch)
        results.update(res)
        jitter = random.uniform(0, JITTER)
        time.sleep(BASE_SLEEP + jitter)
    df = pd.DataFrame(results).T.sort_index()
    return df


def build_variant_id(row):
    # chrom-pos-ref-alt, e.g. "1-123456-A-T"
    return f"{row['chr']}-{row['chr_pos_38']}-{row['ref_nt']}-{row['alt_nt']}"


def _post_with_retries(query: str) -> dict:
    """POST to gnomAD API with retries + backoff."""
    sleep = BASE_SLEEP
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, json={"query": query}, timeout=TIMEOUT_SEC)
            # If rate-limited or server error, treat specially
            if resp.status_code in (429, 500, 502, 503, 504):
                msg = f"HTTP {resp.status_code} on attempt {attempt}"
                if attempt == MAX_RETRIES:
                    raise RuntimeError(msg)
                # backoff
                jitter = random.uniform(0, JITTER)
                time.sleep(sleep + jitter)
                sleep *= BACKOFF_FACTOR
                continue

            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # network issues: retry
            if attempt == MAX_RETRIES:
                raise
            jitter = random.uniform(0, JITTER)
            time.sleep(sleep + jitter)
            sleep *= BACKOFF_FACTOR

    # Should never reach here
    raise RuntimeError("Exhausted retries unexpectedly")


def query_gnomad(variant_ids):
    """
    Query gnomAD for a list of variant IDs.
    Returns a dict: {variant_id: {'ac':..., 'an':..., 'af':...}, ...}
    """
    fields = []
    for i, vid in enumerate(variant_ids):
        alias = f"v{i}"
        fields.append(
            f'{alias}: variant(variantId: "{vid}", dataset: {DATASET}) '
            "{ variantId exome { ac ac_hemi ac_hom an af filters flags} genome { ac ac_hemi ac_hom an af filters flags} joint { ac homozygote_count hemizygote_count an filters} }"
        )
    query = "{ " + " ".join(fields) + " }"

    json_data = _post_with_retries(query)
    data = json_data["data"]

    out = {}
    for i, vid in enumerate(variant_ids):
        alias = f"v{i}"
        v = data.get(alias)
        out[vid] = {
            "ac_exome": None,
            "ac_hemi_exome": None,
            "ac_hom_exome": None,
            "an_exome": None,
            "af_exome": None,
            "filters_exome": None,
            "flags_exome": None,
            "ac_genome": None,
            "ac_hemi_genome": None,
            "ac_hom_genome": None,
            "an_genome": None,
            "af_genome": None,
            "filters_genome": None,
            "flags_genome": None,
            "ac_joint": None,
            "ac_hemi_joint": None,
            "ac_hom_joint": None,
            "an_joint": None,
            "filters_joint": None,
        }
        if v is None:
            continue
        if v.get("exome") is not None:
            out[vid].update(
                {
                    "ac_exome": v["exome"]["ac"],
                    "ac_hemi_exome": v["exome"]["ac_hemi"],
                    "ac_hom_exome": v["exome"]["ac_hom"],
                    "an_exome": v["exome"]["an"],
                    "af_exome": v["exome"]["af"],
                    "filters_exome": v["exome"]["filters"],
                    "flags_exome": v["exome"]["flags"],
                }
            )
        if v.get("genome") is not None:
            out[vid].update(
                {
                    "ac_genome": v["genome"]["ac"],
                    "ac_hemi_genome": v["genome"]["ac_hemi"],
                    "ac_hom_genome": v["genome"]["ac_hom"],
                    "an_genome": v["genome"]["an"],
                    "af_genome": v["genome"]["af"],
                    "filters_genome": v["genome"]["filters"],
                    "flags_genome": v["genome"]["flags"],
                }
            )
        if v.get("joint") is not None:
            out[vid].update(
                {
                    "ac_joint": v["joint"]["ac"],
                    "ac_hemi_joint": v["joint"]["hemizygote_count"],
                    "ac_hom_joint": v["joint"]["homozygote_count"],
                    "an_joint": v["joint"]["an"],
                    "filters_joint": v["joint"]["filters"],
                }
            )
    return out
