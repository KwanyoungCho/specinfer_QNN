#!/usr/bin/env python3
"""
Generate prompt files from Spec-Bench JSONL for benchmarking.

Usage:
    python3 generate_spec_bench_prompts.py                          # all categories
    python3 generate_spec_bench_prompts.py --category translation   # single category
    python3 generate_spec_bench_prompts.py --category translation qa math_reasoning  # multiple
    python3 generate_spec_bench_prompts.py --max-tokens 512         # skip prompts longer than N tokens (approx)
    python3 generate_spec_bench_prompts.py --data-path /path/to/question.jsonl

Spec-Bench repo: https://github.com/hemingkx/Spec-Bench
"""

import argparse
import json
import os
import sys

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "spec-bench-data", "data", "spec_bench", "question.jsonl"
)

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe.  Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)


def format_llama3_prompt(user_message: str) -> str:
    """Format prompt using Llama 3 chat template."""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )


def approx_token_count(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4


def main():
    parser = argparse.ArgumentParser(description="Generate Spec-Bench prompt files")
    parser.add_argument(
        "--data-path", default=DEFAULT_DATA_PATH,
        help="Path to Spec-Bench question.jsonl",
    )
    parser.add_argument(
        "--output-dir", default="spec_bench_prompts",
        help="Output directory for prompt files (default: spec_bench_prompts)",
    )
    parser.add_argument(
        "--category", nargs="*", default=None,
        help="Filter by category (e.g. translation qa math_reasoning). Default: all",
    )
    parser.add_argument(
        "--max-prompt-chars", type=int, default=None,
        help="Skip prompts with more than N characters (helps avoid context overflow)",
    )
    parser.add_argument(
        "--list-categories", action="store_true",
        help="List available categories and exit",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("Download Spec-Bench first:")
        print("  cd ../")
        print("  git clone https://github.com/hemingkx/Spec-Bench.git spec-bench-data")
        sys.exit(1)

    entries = []
    with open(args.data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    if args.list_categories:
        cats = {}
        for e in entries:
            cat = e.get("category", "unknown")
            cats[cat] = cats.get(cat, 0) + 1
        print("Available categories:")
        for cat, count in sorted(cats.items()):
            print(f"  {cat}: {count} prompts")
        print(f"  TOTAL: {sum(cats.values())} prompts")
        return

    if args.category:
        allowed = set(args.category)
        before = len(entries)
        entries = [e for e in entries if e.get("category") in allowed]
        print(f"Filtered by categories {args.category}: {before} -> {len(entries)}")

    os.makedirs(args.output_dir, exist_ok=True)

    generated = 0
    skipped = 0

    for idx, entry in enumerate(entries, start=1):
        question_id = entry.get("question_id", idx)
        category = entry.get("category", "unknown")
        turns = entry.get("turns", [])
        if not turns:
            skipped += 1
            continue

        user_message = turns[0]

        if args.max_prompt_chars and len(user_message) > args.max_prompt_chars:
            skipped += 1
            continue

        prompt = format_llama3_prompt(user_message)
        num = f"{generated + 1:03d}"

        prompt_file = os.path.join(args.output_dir, f"prompt_{num}.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt)

        meta_file = os.path.join(args.output_dir, f"meta_{num}.txt")
        with open(meta_file, "w") as f:
            f.write(f"Category: {category}\n")
            f.write(f"Question ID: {question_id}\n")
            f.write(f"Prompt chars: {len(user_message)}\n")
            f.write(f"Question: {user_message[:200]}\n")

        generated += 1

    print(f"Generated {generated} prompt files in {args.output_dir}/")
    if skipped:
        print(f"Skipped {skipped} prompts (empty or too long)")
    print(f"Prompt files: prompt_001.txt to prompt_{generated:03d}.txt")
    print(f"Metadata files: meta_001.txt to meta_{generated:03d}.txt")


if __name__ == "__main__":
    main()
