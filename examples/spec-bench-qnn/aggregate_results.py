#!/usr/bin/env python3
"""Aggregate spec-bench-qnn per-prompt output files into a CSV summary."""

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_output_file(filepath):
    metrics = {
        "sample_index": None,
        "question_id": None,
        "category": None,
        "status": None,
        "prefill_tokens": None,
        "prefill_ms": None,
        "prefill_tps": None,
        "decode_tokens": None,
        "decode_ms": None,
        "decode_tps": None,
        "decode_lat_ms": None,
        "draft_len": None,
        "accept_len": None,
        "accept_ratio": None,
        "avg_draft_lat_ms": None,
        "avg_verify_lat_ms": None,
        "avg_td_ms": None,
    }

    try:
        content = Path(filepath).read_text()

        sample_match = re.search(r"Sample index\s+:\s+(\d+)", content)
        if sample_match:
            metrics["sample_index"] = int(sample_match.group(1))
        else:
            file_match = re.search(r"output_(\d+)\.txt", str(filepath))
            if file_match:
                metrics["sample_index"] = int(file_match.group(1))

        question_match = re.search(r"Question ID\s+:\s+(-?\d+)", content)
        if question_match:
            metrics["question_id"] = int(question_match.group(1))

        category_match = re.search(r"Category\s+:\s+(.+)", content)
        if category_match:
            metrics["category"] = category_match.group(1).strip()

        status_match = re.search(r"Status\s+:\s+(\w+)", content)
        if status_match:
            metrics["status"] = status_match.group(1).strip().lower()

        prefill_match = re.search(
            r"Prefill\s+:\s+(\d+)\s+tokens\s+\|\s+([\d.]+)\s+ms\s+\|\s+([\d.]+)\s+t/s",
            content,
        )
        if prefill_match:
            metrics["prefill_tokens"] = int(prefill_match.group(1))
            metrics["prefill_ms"] = float(prefill_match.group(2))
            metrics["prefill_tps"] = float(prefill_match.group(3))

        decode_match = re.search(
            r"Decode\s+:\s+(\d+)\s+tokens\s+\|\s+([\d.]+)\s+ms\s+\|\s+([\d.]+)\s+t/s",
            content,
        )
        if decode_match:
            metrics["decode_tokens"] = int(decode_match.group(1))
            metrics["decode_ms"] = float(decode_match.group(2))
            metrics["decode_tps"] = float(decode_match.group(3))

        decode_lat_match = re.search(r"Decode latency\s+:\s+\|\s+([\d.]+)\s+ms/tok", content)
        if decode_lat_match:
            metrics["decode_lat_ms"] = float(decode_lat_match.group(1))

        draft_len_match = re.search(r"Draft length\s+:\s+([\d.]+)", content)
        if draft_len_match:
            metrics["draft_len"] = float(draft_len_match.group(1))

        accept_len_match = re.search(r"Avg accept length\s+:\s+([\d.]+)", content)
        if accept_len_match:
            metrics["accept_len"] = float(accept_len_match.group(1))

        accept_ratio_match = re.search(r"Accept ratio\s+:\s+([\d.]+)%", content)
        if accept_ratio_match:
            metrics["accept_ratio"] = float(accept_ratio_match.group(1))

        avg_draft_match = re.search(r"Avg draft phase\s+:\s+([\d.]+)\s+ms", content)
        if avg_draft_match:
            metrics["avg_draft_lat_ms"] = float(avg_draft_match.group(1))

        avg_verify_match = re.search(r"Avg verification\s+:\s+([\d.]+)\s+ms", content)
        if avg_verify_match:
            metrics["avg_verify_lat_ms"] = float(avg_verify_match.group(1))

        avg_td_match = re.search(r"Avg T_d \(1-tok dft\)\s+:\s+([\d.]+)\s+ms", content)
        if avg_td_match:
            metrics["avg_td_ms"] = float(avg_td_match.group(1))

    except Exception as exc:
        print(f"Warning: failed to parse {filepath}: {exc}", file=sys.stderr)

    return metrics


def average(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return (sum(values) / len(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description="Aggregate spec-bench-qnn output files")
    parser.add_argument("output_dir", help="Directory containing output_*.txt files")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the aggregate CSV file (default: <output_dir>/<name>_results.csv)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    output_files = sorted(output_dir.glob("output_*.txt"))
    if not output_files:
        print(f"Error: no output_*.txt files found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    rows = [parse_output_file(path) for path in output_files]
    rows = [row for row in rows if row["sample_index"] is not None]
    rows.sort(key=lambda row: row["sample_index"])

    csv_path = Path(args.output) if args.output else output_dir / f"{output_dir.name}_results.csv"

    fieldnames = [
        "sample_index",
        "question_id",
        "category",
        "status",
        "prefill_tokens",
        "prefill_ms",
        "prefill_tps",
        "decode_tokens",
        "decode_ms",
        "decode_tps",
        "decode_lat_ms",
        "draft_len",
        "accept_len",
        "accept_ratio",
        "avg_draft_lat_ms",
        "avg_verify_lat_ms",
        "avg_td_ms",
    ]

    success_rows = [row for row in rows if row.get("status") == "success"]
    avg_row = {"sample_index": "AVERAGE", "status": "success_only"}
    for field in fieldnames[1:]:
        if field in {"category", "status"}:
            continue
        avg_row[field] = average(success_rows, field)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(avg_row)

    print(f"Found {len(rows)} output files")
    print(f"Saved aggregate CSV to: {csv_path}")

    if success_rows:
        print("\n============================================================")
        print(f"  Summary Statistics ({len(success_rows)} successful prompts)")
        print("============================================================")
        if avg_row["prefill_ms"] is not None:
            print(f"  Avg Prefill:        {avg_row['prefill_ms']:.2f} ms  |  {avg_row['prefill_tps']:.2f} t/s")
        if avg_row["decode_ms"] is not None:
            print(f"  Avg Decode:         {avg_row['decode_ms']:.2f} ms  |  {avg_row['decode_tps']:.2f} t/s")
        if avg_row["decode_lat_ms"] is not None:
            print(f"  Avg Decode Latency: {avg_row['decode_lat_ms']:.2f} ms/tok")
        print("------------------------------------------------------------")
        if avg_row["draft_len"] is not None:
            print(f"  Avg Draft Length:   {avg_row['draft_len']:.3f}")
        if avg_row["accept_len"] is not None:
            print(f"  Avg Accept Length:  {avg_row['accept_len']:.3f}")
        if avg_row["accept_ratio"] is not None:
            print(f"  Avg Accept Ratio:   {avg_row['accept_ratio']:.2f}%")
        print("------------------------------------------------------------")
        if avg_row["avg_draft_lat_ms"] is not None:
            print(f"  Avg Draft Phase:    {avg_row['avg_draft_lat_ms']:.3f} ms")
        if avg_row["avg_verify_lat_ms"] is not None:
            print(f"  Avg Verification:   {avg_row['avg_verify_lat_ms']:.3f} ms")
        if avg_row["avg_td_ms"] is not None:
            print(f"  Avg T_d:            {avg_row['avg_td_ms']:.3f} ms")
        print("============================================================")


if __name__ == "__main__":
    main()
