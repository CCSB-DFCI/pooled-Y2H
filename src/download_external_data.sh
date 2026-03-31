#!/bin/bash
# TODO: check this script actually works

# TODO: this will fail if you run it from another directory
cd ../data/external/

wget https://zenodo.org/records/10813168/files/AlphaMissense_aa_substitutions.tsv.gz
gunzip AlphaMissense_aa_substitutions.tsv.gz

wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.dat.gz
gunzip UP000005640_9606.dat.gz

# 2025_03 is the version of uniprot used by AlphaFoldDB Sep 2025 (v6) release 
wget https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2025_03/knowledgebase/uniprot_sprot-only2025_03.tar.gz
tar -xzf uniprot_sprot-only2025_03.tar.gz
gunzip uniprot_sprot.dat.gz
gunzip uniprot_sprot.fasta.gz
gunzip uniprot_sprot.xml.gz
gunzip uniprot_sprot_varsplic.fasta.gz
rm uniprot_sprot-only2025_03.tar.gz

# TODO: version this file properly
# at the moment I just put the date I downloaded it
# but the last modified date is eariler than that (2024-03-31)
# and there are more recent release versions of clinvar, but not for that file...
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
gunzip variant_summary.txt.gz
mv variant_summary.txt clinvar_variant_summary_2025-08-10.txt

wget https://purl.obolibrary.org/obo/mondo.obo

wget https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv

wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw

wget https://zenodo.org/records/18511521/files/mavedb-dump.20260206153444.zip

wget https://search.thegencc.org/download/action/submissions-export-tsv

# Don't currently use this:
wget https://raw.githubusercontent.com/patterninstitute/grantham/refs/heads/main/data-raw/grantham_distance_matrix.csv

wget https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz
gunzip pdb_chain_uniprot.tsv.gz

#tissue expression stuff
wget https://v25.proteinatlas.org/download/tsv/normal_ihc_data.tsv.zip  #sha1sum 27c1e90a03fb1e7d5b4653c2b6980a14f3f04a1c ; downloaded Jan 2026
unzip normal_ihc_data.tsv.zip

wget https://data.monarchinitiative.org/monarch-kg/2026-01-11/tsv/all_associations/disease_or_phenotypic_feature_to_location_association.all.tsv.gz
gunzip disease_or_phenotypic_feature_to_location_association.all.tsv.gz

wget https://ftp.ebi.ac.uk/pub/databases/intact/2026-01-09/psimitab/features/mutations.tsv
mv mutations.tsv intact_mutations_2026-01-09.tsv

# Sahni et al. Cell 2015 Table S3 from here:
# https://www.sciencedirect.com/science/article/pii/S0092867415004304
# TODO: probably need to check it into the repo
