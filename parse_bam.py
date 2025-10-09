#!/usr/bin/env python3
import pysam
import csv
import argparse
from pathlib import Path
from collections import Counter

parser = argparse.ArgumentParser(description="Parse BAM and export key info.")
parser.add_argument("--input", "-i", required=True, help="Input BAM file")
parser.add_argument("--out-prefix", "-o", required=True, help="Output prefix (no extension)")
parser.add_argument("--with-seq", action="store_true", help="Also export read sequence (large files)")
args = parser.parse_args()

bam_path = Path(args.input)
out_prefix = Path(args.out_prefix)

reads_tsv = out_prefix.with_suffix(".reads.tsv")
summary_tsv = out_prefix.with_suffix(".summary.tsv")
refcounts_tsv = out_prefix.with_suffix(".refcounts.tsv")

with pysam.AlignmentFile(bam_path, "rb") as bam, open(reads_tsv, "w", newline="") as rf:
    cols = [
        "read_name","reference_name","position","reference_end","mapq",
        "cigar","is_reverse","is_read1","is_read2","is_proper_pair",
        "is_secondary","is_supplementary","is_duplicate","is_unmapped","read_length"
    ]
    if args.with_seq:
        cols.append("query_sequence")
    writer = csv.writer(rf, delimiter="\t")
    writer.writerow(cols)

    counts = Counter()
    per_ref = Counter()

    for read in bam.fetch(until_eof=True):
        counts["total"] += 1
        if read.is_unmapped:
            counts["unmapped"] += 1
        else:
            counts["mapped"] += 1
            per_ref[read.reference_name] += 1
        if read.is_duplicate:
            counts["duplicates"] += 1

        row = [
            read.query_name, read.reference_name, read.reference_start, read.reference_end,
            read.mapping_quality, read.cigarstring, read.is_reverse, read.is_read1,
            read.is_read2, read.is_proper_pair, read.is_secondary, read.is_supplementary,
            read.is_duplicate, read.is_unmapped, read.query_length
        ]
        if args.with_seq:
            row.append(read.query_sequence)
        writer.writerow(row)

# write summary
with open(summary_tsv, "w", newline="") as s:
    w = csv.writer(s, delimiter="\t")
    w.writerow(["metric","value"])
    for k,v in counts.items():
        w.writerow([k,v])
    if counts["total"] > 0:
        w.writerow(["mapped_percent", f"{100*counts['mapped']/counts['total']:.2f}"])

# write per-reference counts
with open(refcounts_tsv, "w", newline="") as r:
    w = csv.writer(r, delimiter="\t")
    w.writerow(["reference_name","read_count"])
    for ref,cnt in sorted(per_ref.items(), key=lambda x: -x[1]):
        w.writerow([ref,cnt])

print(f"✅ Parsing finished.\nReads: {reads_tsv}\nSummary: {summary_tsv}\nRef counts: {refcounts_tsv}")

