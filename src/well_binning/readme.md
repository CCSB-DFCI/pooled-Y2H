# bin_ont_reads.py
A script to bin Oxford Nanopore `fastq.gz` reads by well using forward <ins>and</ins> reverse barcodes. Input is one `fastq.gz` file. Output is one subsequence `fastq.gz` file for each well.

:warning: NOTE: when well-binning samples from experiment `HsVcPPIP01v6AD`, use the script `bin_ont_reads_adc.py` with the `--vector` parameter set to `adc`

Required non-python libraries: 
+ zlib ([v1.2.11](https://zlib.net/fossils/zlib-1.2.11.tar.gz) was used for well-binning)

## <b>Before starting:</b>
install `itersplit_fastq`:

```
sudo make install
```

## <b> Command Line Arguments</b>
```
python3 bin_ont_reads.py --help

options:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        Input ONT fastq.gz file
  --cpus CPUS           n of CPUs to use. -1 to use all available
  --max-errors-primer MAX_ERRORS_PRIMER
                        maximum allowable errors in primer matching [adapter - [ATCG]{13} - vector sequence]. Can also mean [adapter - [ATCG]{13}] or [[ATCG]{13} - vector sequence]. Allowing more than 1 error makes pattern matching slower
  --max-errors-barcode {0,1,2,3,4,5}
                        maximum allowable errors in barcode matching [0-5]
  --barcode-type {m13,swim}
                        barcode type [m13, swim]. If m13, currently only M13G_for_i7 and M13G_rev_i5 are supported. If swim, --vector must be specified (ad or db)
  --vector {ad,db}      If --barcode-type=swim, must specifiy whether ad or db
  --for-set-id FOR_SET_ID
                        Set ID of forward primer (e.g. 'J_AD'). Must exactly match a value from the set_id column of the kiloseq primers table
  --for-primer-type FOR_PRIMER_TYPE
                        Type of forward primer (e.g. 'DB_n_for'). Must exactly match a value from the primer_type column of the kiloseq primers table
  --rev-set-id REV_SET_ID
                        Set ID of reverse primer (e.g. 'D_AD'). Must exactly match a value from the set_id column of the kiloseq primers table
  --rev-primer-type REV_PRIMER_TYPE
                        Type of reverse primer (e.g. 'term_i5'). Must exactly match a value from the primer_type column of the kiloseq primers table

```


<b>Example usage: </b>
<i>(DB forward idx "J" + TERM reverse idx "D") using all CPUs:</i>

```
python3 bin_ont_reads.py --input-file <fastq.gz file> --cpus -1 --max-errors-primer 1 --max-errors-barcode 1 --barcode-type swim --vector db --for-set-id J_DB --for-primer-type DB_n_for --rev-set-id D_AD --rev-primer-type term_i5
```

## <b>Output:</b>
  1) well-specific subsequence ```fastq``` files in the ```well_subseq``` folder
  2) excel table showing per-well read counts
  3) a read-well mapping python dictionary
  4) a pie chart showing various categories reads can fall under
  
#### Categories: 
+ <ins>Mapped by one barcode:</ins> n of reads that were unambigusouly mapped to a well using only one barcode <b>[reads retained]</b>
+ <ins>Mapped by two barcodes:</ins> n of reads that were unambiguously mapped to a well using two barcodes <b>[reads retained]</b>
+ <ins>Barcode conflict:</ins> n of reads with two observed barcodes that match expected barcodes but reference different wells <b>[reads discarded]</b>
+ <ins>Observed barcode no match:</ins> n of reads with a single barcode that doesn't match any expected barcode <b>[reads discarded]</b>
+ <ins>Multiple primer patterns:</ins> n of reads with multiple primer sequence matches <b>[reads discarded]</b>
+ <ins>Barcode ambiguous</ins>: n of reads with one or two observed barcodes and the barcode(s) doesn't/don't match an expected barcode, or the observed barcode(s) match(es) multiple expected barcodes. Also includes any reads that were uncategorized after all processing steps <b>[reads discarded]</b>

## <b>Case Handling</b>

For a read to be mapped to a well:
+ if two observed barcodes on a read match expected barcodes, both barcodes must reference the same well, else the read is discarded
+ at least one observed barcode must map to a barcode in the expected space
+ only one `adapter - [ATCG]{13} - vector seq` forward or reverse primer pattern can be present in the read. If >1 matches for a single primer are found, the read is discarded
+ an observed barcode can only match a single expected barcode, else the read is discarded
+ if both forward and reverse primer sequences are found in a read, the strandedness must match what's expected, else the read is discarded. For example, if a forward primer sequence is found, then the only valid reverse primer sequence that can be present is the reverse primer's reverse complement 

## <b> Performance </b> 

- A ~6 gigabyte `fastq.gz` file with ~4M reads took ~4.5 min to run using 64 CPUs




