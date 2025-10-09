#!/usr/bin/env python3
import pysam
import csv
import gzip
from collections import Counter

# Input BAM file
input_bam = "/vol/projects/znoor/2024_08_16_090_pl4_1004517201_H6_S90_L001.sorted.bam"


# Output files
positions_summary = "/vol/projects/znoor/psivapor_positions_summary.tsv"
coverage_out = "/vol/projects/znoor/psivapor_coverage.tsv.gz"
reads_out = "/vol/projects/znoor/psivapor_reads_per_position.tsv.gz"
bases_out = "/vol/projects/znoor/psivapor_bases_per_position.tsv.gz"

# Open BAM
bamfile = pysam.AlignmentFile(input_bam, "rb")

# 1️⃣ Task 1: Count reference positions
with open(positions_summary, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["reference_name", "reference_length"])
    total_len = 0
    for ref in bamfile.references:
        ref_len = bamfile.get_reference_length(ref)
        total_len += ref_len
        writer.writerow([ref, ref_len])
    writer.writerow(["TOTAL", total_len])
print(f"✅ (1) Reference positions written to {positions_summary}")

# 2️⃣, 3️⃣, 4️⃣ Task: coverage, read IDs, and base-level info
with gzip.open(coverage_out, "wt", newline="") as cov_f, \
     gzip.open(reads_out, "wt", newline="") as reads_f, \
     gzip.open(bases_out, "wt", newline="") as bases_f:

    cov_writer = csv.writer(cov_f, delimiter="\t")
    reads_writer = csv.writer(reads_f, delimiter="\t")
    bases_writer = csv.writer(bases_f, delimiter="\t")

    cov_writer.writerow(["reference_name", "position", "depth", "A", "C", "G", "T", "deletions", "refskips"])
    reads_writer.writerow(["reference_name", "position", "read_name"])
    bases_writer.writerow([
        "reference_name", "position", "read_name",
        "base", "base_quality", "mapping_quality",
        "is_del", "is_refskip", "is_reverse",
        "is_read1", "is_read2", "read_length"
    ])

    # Loop over all positions in BAM using pileup
    for ref in bamfile.references:
        print(f"🔍 Processing {ref} ...")
        for column in bamfile.pileup(ref, stepper="all", max_depth=100000):
            pos = column.pos + 1  # 1-based position
            depth = column.nsegments
            base_counts = Counter()
            del_count = 0
            refskip_count = 0

            for pileup_read in column.pileups:
                aln = pileup_read.alignment
                reads_writer.writerow([ref, pos, aln.query_name])

                if pileup_read.is_del:
                    del_count += 1
                    bases_writer.writerow([ref, pos, aln.query_name, "-", None, aln.mapping_quality, True, False, aln.is_reverse, aln.is_read1, aln.is_read2, aln.query_length])
                    continue
                if pileup_read.is_refskip:
                    refskip_count += 1
                    bases_writer.writerow([ref, pos, aln.query_name, ">", None, aln.mapping_quality, False, True, aln.is_reverse, aln.is_read1, aln.is_read2, aln.query_length])
                    continue

                qpos = pileup_read.query_position
                if qpos is not None and aln.query_sequence:
                    base = aln.query_sequence[qpos]
                    qual = aln.query_qualities[qpos] if aln.query_qualities else None
                    base_counts[base.upper()] += 1
                    bases_writer.writerow([ref, pos, aln.query_name, base, qual, aln.mapping_quality, False, False, aln.is_reverse, aln.is_read1, aln.is_read2, aln.query_length])

            cov_writer.writerow([
                ref, pos, depth,
                base_counts.get("A", 0),
                base_counts.get("C", 0),
                base_counts.get("G", 0),
                base_counts.get("T", 0),
                del_count, refskip_count
            ])

bamfile.close()
print("🎉 Done! All four tasks completed successfully.")

