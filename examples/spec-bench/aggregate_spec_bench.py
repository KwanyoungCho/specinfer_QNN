#!/usr/bin/env python3
"""
Aggregate Spec-Bench results.

Works with two modes:
  1. CSV from C++ binary:  python3 aggregate_spec_bench.py results.csv
  2. Output dir from shell: python3 aggregate_spec_bench.py output_dir/ --meta-dir prompts_dir/

The C++ binary (llama-spec-bench) already writes a CSV, so this script is
mainly useful for re-analysis, filtering, or pretty-printing.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_output_file(filepath):
    """Parse a single output file (from shell script runner) and extract metrics."""
    metrics = {}
    try:
        content = Path(filepath).read_text()

        m = re.search(r'output_(\d+)\.txt', str(filepath))
        if m:
            metrics['question_id'] = int(m.group(1))

        m = re.search(r'Prefill\s+:\s+(\d+)\s+tokens\s+\|\s+([\d.]+)\s+ms\s+\|\s+([\d.]+)\s+t/s', content)
        if m:
            metrics['n_input'] = int(m.group(1))
            metrics['prefill_ms'] = float(m.group(2))

        m = re.search(r'Decode\s+:\s+(\d+)\s+tokens\s+\|\s+([\d.]+)\s+ms\s+\|\s+([\d.]+)\s+t/s', content)
        if m:
            metrics['n_predict'] = int(m.group(1))
            metrics['decode_ms'] = float(m.group(2))
            metrics['decode_tps'] = float(m.group(3))

        m = re.search(r'Decode latency\s+:\s+\|\s+([\d.]+)\s+ms/tok', content)
        if m:
            metrics['decode_lat_ms'] = float(m.group(1))

        m = re.search(r'Avg accept length\s+:\s+([\d.]+)', content)
        if m:
            metrics['accept_len'] = float(m.group(1))

        m = re.search(r'Accept ratio\s+:\s+([\d.]+)%', content)
        if m:
            metrics['accept_ratio'] = float(m.group(1))

        m = re.search(r'Avg draft phase\s+:\s+([\d.]+)\s+ms', content)
        if m:
            metrics['avg_draft_ms'] = float(m.group(1))

        m = re.search(r'Avg verification\s+:\s+([\d.]+)\s+ms', content)
        if m:
            metrics['avg_verify_ms'] = float(m.group(1))

        m = re.search(r'Avg T_d \(1-tok dft\)\s+:\s+([\d.]+)\s+ms', content)
        if m:
            metrics['avg_td_ms'] = float(m.group(1))

    except Exception as e:
        print(f"Warning: error parsing {filepath}: {e}", file=sys.stderr)

    return metrics


def load_from_csv(path):
    """Load results from a CSV file (produced by llama-spec-bench)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_from_output_dir(output_dir, meta_dir=None):
    """Load results from per-prompt output files (shell script runner)."""
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("output_*.txt"))
    rows = []
    for fp in files:
        m = parse_output_file(fp)
        if not m:
            continue
        # Try to read category from meta file
        num = re.search(r'output_(\d+)\.txt', fp.name)
        if num and meta_dir:
            meta_path = Path(meta_dir) / f"meta_{num.group(1)}.txt"
            if meta_path.exists():
                for line in meta_path.read_text().splitlines():
                    if line.startswith("Category:"):
                        m['category'] = line.split(":", 1)[1].strip()
                        break
        rows.append(m)
    return rows


def print_summary(rows, key_tps='decode_tps', key_al='accept_len', key_ar='accept_ratio', key_lat='decode_lat_ms'):
    """Print per-category and overall summary."""
    by_cat = defaultdict(list)
    for r in rows:
        cat = r.get('category', 'unknown')
        by_cat[cat].append(r)

    def avg(lst, key):
        vals = [float(x[key]) for x in lst if key in x and x[key] not in (None, '', 'None')]
        return sum(vals) / len(vals) if vals else 0

    print(f"\n{'='*70}")
    print(f"  Spec-Bench Summary ({len(rows)} prompts)")
    print(f"{'='*70}")
    print(f"  {'Category':<20s} {'N':>4s} {'t/s':>8s} {'accept_len':>11s} {'accept_%':>9s} {'lat(ms)':>8s}")
    print(f"  {'-'*20} {'-'*4} {'-'*8} {'-'*11} {'-'*9} {'-'*8}")

    for cat in sorted(by_cat.keys()):
        grp = by_cat[cat]
        print(f"  {cat:<20s} {len(grp):>4d} {avg(grp, key_tps):>8.2f} {avg(grp, key_al):>11.2f} {avg(grp, key_ar):>8.1f}% {avg(grp, key_lat):>8.2f}")

    print(f"  {'-'*20} {'-'*4} {'-'*8} {'-'*11} {'-'*9} {'-'*8}")
    print(f"  {'OVERALL':<20s} {len(rows):>4d} {avg(rows, key_tps):>8.2f} {avg(rows, key_al):>11.2f} {avg(rows, key_ar):>8.1f}% {avg(rows, key_lat):>8.2f}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Spec-Bench results")
    parser.add_argument("input", help="CSV file or output directory")
    parser.add_argument("--meta-dir", default=None, help="Directory with meta_NNN.txt files (for shell runner mode)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix == '.csv':
        rows = load_from_csv(input_path)
        print_summary(rows)
    elif input_path.is_dir():
        rows = load_from_output_dir(input_path, args.meta_dir)
        if not rows:
            print(f"No output files found in {input_path}", file=sys.stderr)
            return
        print_summary(rows, key_tps='decode_tps', key_al='accept_len', key_ar='accept_ratio', key_lat='decode_lat_ms')
    else:
        print(f"Error: {args.input} is not a CSV file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
