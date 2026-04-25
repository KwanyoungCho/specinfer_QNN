#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict


OVERALL = "OVERALL"


@dataclass
class TokenAggregate:
    token_id: int
    token_text: str = ""
    verified_total: int = 0
    accepted_total: int = 0
    bonus_total: int = 0
    proposed_total: int = 0
    target_generated_total: int = 0
    verified_pos: dict[int, int] = field(default_factory=dict)
    accepted_pos: dict[int, int] = field(default_factory=dict)
    bonus_pos: dict[int, int] = field(default_factory=dict)
    proposed_pos: dict[int, int] = field(default_factory=dict)


def get_token_bucket(
    stats_by_category: DefaultDict[str, dict[int, TokenAggregate]],
    category: str,
    token_id: int,
    token_text: str = "",
) -> TokenAggregate:
    bucket = stats_by_category[category]
    if token_id not in bucket:
        bucket[token_id] = TokenAggregate(token_id=token_id, token_text=token_text)
    agg = bucket[token_id]
    if token_text and not agg.token_text:
        agg.token_text = token_text
    return agg


def add_pos(dst: dict[int, int], position: int, count: int) -> None:
    dst[position] = dst.get(position, 0) + count


def load_token_pos_stats(path: Path) -> DefaultDict[str, dict[int, TokenAggregate]]:
    stats_by_category: DefaultDict[str, dict[int, TokenAggregate]] = defaultdict(dict)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            token_id = int(row["token_id"])
            token_text = row["token_text"]
            position = int(row["position"])
            verified = int(row["verified_count"])
            accepted = int(row["accepted_count"])
            bonus = int(row["bonus_count"])
            proposed = int(row["proposed_count"])

            agg = get_token_bucket(stats_by_category, category, token_id, token_text)
            agg.verified_total += verified
            agg.accepted_total += accepted
            agg.bonus_total += bonus
            agg.proposed_total += proposed
            if verified:
                add_pos(agg.verified_pos, position, verified)
            if accepted:
                add_pos(agg.accepted_pos, position, accepted)
            if bonus:
                add_pos(agg.bonus_pos, position, bonus)
            if proposed:
                add_pos(agg.proposed_pos, position, proposed)
    return stats_by_category


def load_accept_hist(path: Path | None) -> DefaultDict[str, dict[str, dict[int, int]]]:
    hist: DefaultDict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    if path is None:
        return hist

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            kind = row["kind"]
            length = int(row["length"])
            count = int(row["count"])
            hist[category][kind][length] = hist[category][kind].get(length, 0) + count
    return hist


def load_target_generated_freq(
    path: Path | None,
    stats_by_category: DefaultDict[str, dict[int, TokenAggregate]],
) -> None:
    if path is None:
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            token_id = int(row["token_id"])
            token_text = row["token_text"]
            count = int(row["count"])
            agg = get_token_bucket(stats_by_category, category, token_id, token_text)
            agg.target_generated_total += count


def load_token_freq(
    path: Path | None,
    stats_by_category: DefaultDict[str, dict[int, TokenAggregate]],
) -> None:
    if path is None:
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_id = int(row["token_id"])
            token_text = row["token_text"]
            draft_count = int(row["draft_count"])
            accepted_count = int(row["accepted_count"])
            bonus_count = int(row["bonus_count"])

            agg = get_token_bucket(stats_by_category, OVERALL, token_id, token_text)
            agg.proposed_total = max(agg.proposed_total, draft_count)
            agg.accepted_total = max(agg.accepted_total, accepted_count)
            agg.bonus_total = max(agg.bonus_total, bonus_count)


def sanitize_category_filename(category: str) -> str:
    safe = []
    for ch in category:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "category"


def max_position_for_source(agg: TokenAggregate, source: str) -> int:
    pos_map = agg.verified_pos if source == "verified" else agg.accepted_pos
    return max(pos_map.keys(), default=0)


def total_from_mode(agg: TokenAggregate, mode: str, freq_source: str) -> float:
    if mode == "freq":
        if freq_source == "target-generated":
            return float(agg.target_generated_total)
        return float(agg.verified_total)
    if mode == "accepted":
        return float(agg.accepted_total)
    raise ValueError(f"unsupported total mode: {mode}")


def survival_probability(hist: dict[int, int], threshold: int) -> float:
    total = sum(hist.values())
    if total <= 0:
        return 0.0
    if threshold <= 0:
        return 1.0
    survived = sum(count for length, count in hist.items() if length >= threshold)
    return survived / total


def survival_weight(
    position: int,
    source: str,
    survival_kind: str,
    hist_by_kind: dict[str, dict[int, int]],
) -> float:
    if survival_kind == "step_output":
        return survival_probability(hist_by_kind.get("step_output", {}), position)

    threshold = position if source == "accepted" else position - 1
    return survival_probability(hist_by_kind.get("accepted_prefix", {}), threshold)


def prefix_linear_score(agg: TokenAggregate, source: str, max_pos: int) -> float:
    pos_map = agg.verified_pos if source == "verified" else agg.accepted_pos
    score = 0.0
    for position, count in pos_map.items():
        if position > max_pos:
            continue
        score += float(max_pos - position + 1) * count
    return score


def prefix_survival_score(
    agg: TokenAggregate,
    source: str,
    survival_kind: str,
    hist_by_kind: dict[str, dict[int, int]],
    max_pos: int,
) -> float:
    pos_map = agg.verified_pos if source == "verified" else agg.accepted_pos
    score = 0.0
    for position, count in pos_map.items():
        if position > max_pos:
            continue
        score += survival_weight(position, source, survival_kind, hist_by_kind) * count
    return score


def score_token(
    agg: TokenAggregate,
    args: argparse.Namespace,
    hist_by_kind: dict[str, dict[int, int]],
    max_pos: int,
) -> float:
    if args.mode == "freq":
        return total_from_mode(agg, "freq", args.freq_source)
    if args.mode == "accepted":
        return total_from_mode(agg, "accepted", args.freq_source)
    if args.mode == "draft_accept_bonus":
        return (
            args.alpha * agg.proposed_total
            + args.beta * agg.accepted_total
            + args.gamma * agg.bonus_total
        )
    if args.mode == "prefix_linear":
        return prefix_linear_score(agg, args.prefix_count_source, max_pos)
    if args.mode == "prefix_survival":
        return prefix_survival_score(
            agg,
            args.prefix_count_source,
            args.survival_kind,
            hist_by_kind,
            max_pos,
        )
    raise ValueError(f"unsupported mode: {args.mode}")


def choose_hist(
    accept_hist: DefaultDict[str, dict[str, dict[int, int]]],
    category: str,
) -> dict[str, dict[int, int]]:
    if category in accept_hist:
        return accept_hist[category]
    return accept_hist.get(OVERALL, {})


def iter_categories(
    stats_by_category: DefaultDict[str, dict[int, TokenAggregate]],
) -> list[str]:
    categories = sorted(stats_by_category.keys())
    if OVERALL in categories:
        categories.remove(OVERALL)
        categories.insert(0, OVERALL)
    return categories


def write_shortlist(path: Path, token_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted(token_ids), f, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build global/category token shortlists from Spec-Bench raw position stats."
    )
    parser.add_argument("--token-pos-stats", required=True, type=Path)
    parser.add_argument("--accept-hist", type=Path)
    parser.add_argument("--target-generated-freq", type=Path)
    parser.add_argument("--token-freq", type=Path)
    parser.add_argument("--results", type=Path, help="Currently unused, accepted for workflow compatibility")
    parser.add_argument("--mode", required=True, choices=[
        "freq",
        "accepted",
        "draft_accept_bonus",
        "prefix_linear",
        "prefix_survival",
    ])
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--output-dir", default=Path("."), type=Path)
    parser.add_argument("--freq-source", default="verified", choices=["verified", "target-generated"])
    parser.add_argument("--prefix-count-source", default="verified", choices=["verified", "accepted"])
    parser.add_argument("--survival-kind", default="step_output", choices=["accepted_prefix", "step_output"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-position", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats_by_category = load_token_pos_stats(args.token_pos_stats)
    load_target_generated_freq(args.target_generated_freq, stats_by_category)
    load_token_freq(args.token_freq, stats_by_category)
    accept_hist = load_accept_hist(args.accept_hist)

    if args.mode == "freq" and args.freq_source == "target-generated" and not args.target_generated_freq:
        raise SystemExit("--freq-source target-generated requires --target-generated-freq")

    if args.mode == "prefix_survival" and not args.accept_hist:
        raise SystemExit("--mode prefix_survival requires --accept-hist")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shortlist_by_category_dir = output_dir / "shortlist_by_category"
    shortlist_by_category_dir.mkdir(parents=True, exist_ok=True)

    categories = iter_categories(stats_by_category)
    all_score_rows: list[dict[str, object]] = []

    for category in categories:
        token_map = stats_by_category[category]
        hist_by_kind = choose_hist(accept_hist, category)

        observed_max_pos = 0
        for agg in token_map.values():
            observed_max_pos = max(
                observed_max_pos,
                max_position_for_source(agg, args.prefix_count_source),
            )
        max_pos = args.max_position if args.max_position > 0 else observed_max_pos

        scored_rows = []
        for agg in token_map.values():
            score = score_token(agg, args, hist_by_kind, max_pos)
            scored_rows.append({
                "category": category,
                "token_id": agg.token_id,
                "token_text": agg.token_text,
                "score": score,
                "verified_count": agg.verified_total,
                "accepted_count": agg.accepted_total,
                "bonus_count": agg.bonus_total,
                "proposed_count": agg.proposed_total,
            })

        scored_rows.sort(
            key=lambda row: (
                -float(row["score"]),
                -int(row["verified_count"]),
                int(row["token_id"]),
            )
        )
        all_score_rows.extend(scored_rows)

        shortlist_token_ids = [int(row["token_id"]) for row in scored_rows[: args.top_k]]
        if category == OVERALL:
            write_shortlist(output_dir / "shortlist_global_topK.json", shortlist_token_ids)
        else:
            filename = f"{sanitize_category_filename(category)}_topK.json"
            write_shortlist(shortlist_by_category_dir / filename, shortlist_token_ids)

    score_path = output_dir / "shortlist_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "token_id",
                "token_text",
                "score",
                "verified_count",
                "accepted_count",
                "bonus_count",
                "proposed_count",
            ],
        )
        writer.writeheader()
        for row in all_score_rows:
            row = dict(row)
            row["score"] = f"{float(row['score']):.10f}"
            writer.writerow(row)

    print(f"Saved shortlist scores to {score_path}")
    print(f"Saved global shortlist to {output_dir / 'shortlist_global_topK.json'}")
    if len(categories) > 1:
        print(f"Saved category shortlists under {shortlist_by_category_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
