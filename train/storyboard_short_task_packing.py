from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_SHORT_TASK_TYPES = (
    "single_shot",
    "multifield",
    "ordering",
    "summary_compress",
)
DEFAULT_PACKING_SPLITS = ("train",)
KNOWN_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PackGroup:
    task_type: str
    sample_indices: tuple[int, ...]
    total_length: int


def normalize_string_tuple(value: object, fallback: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(fallback)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        normalized = tuple(item for item in items if item)
        return normalized or tuple(fallback)
    if isinstance(value, Iterable):
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        return normalized or tuple(fallback)
    return tuple(fallback)


def infer_split_from_paths(paths: str | Path | Sequence[str | Path]) -> str:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    matched_splits: set[str] = set()
    for path in paths:
        stem = Path(path).stem.lower()
        for split_name in KNOWN_SPLITS:
            if stem == split_name or stem.startswith(f"{split_name}_") or stem.endswith(f"_{split_name}"):
                matched_splits.add(split_name)
    if len(matched_splits) == 1:
        return next(iter(matched_splits))
    return "unknown"


def first_fit_decreasing_pack(
    task_type: str,
    items: Sequence[tuple[int, int]],
    max_length: int,
    max_samples_per_pack: int = 0,
) -> list[PackGroup]:
    bins: list[dict[str, object]] = []
    sorted_items = sorted(items, key=lambda item: item[1], reverse=True)

    for sample_index, sample_length in sorted_items:
        placed = False
        for current_bin in bins:
            bin_total_length = int(current_bin["total_length"])
            bin_sample_indices = current_bin["sample_indices"]
            if bin_total_length + sample_length > max_length:
                continue
            if max_samples_per_pack > 0 and len(bin_sample_indices) >= max_samples_per_pack:
                continue
            bin_sample_indices.append(sample_index)
            current_bin["total_length"] = bin_total_length + sample_length
            placed = True
            break

        if placed:
            continue

        bins.append(
            {
                "sample_indices": [sample_index],
                "total_length": sample_length,
            }
        )

    groups: list[PackGroup] = []
    for current_bin in bins:
        sample_indices = tuple(sorted(int(index) for index in current_bin["sample_indices"]))
        groups.append(
            PackGroup(
                task_type=task_type,
                sample_indices=sample_indices,
                total_length=int(current_bin["total_length"]),
            )
        )
    return groups


def summarize_pack_groups(
    raw_sample_count: int,
    groups: Sequence[PackGroup],
    max_length: int,
    too_long_count: int = 0,
) -> dict[str, float | int]:
    if not groups:
        return {
            "raw_samples": raw_sample_count,
            "packed_items": 0,
            "saved_items": 0,
            "too_long_singletons": too_long_count,
            "multi_sample_packs": 0,
            "avg_pack_size": 0.0,
            "max_pack_size": 0,
            "avg_pack_fill_ratio": 0.0,
            "max_pack_length": 0,
        }

    pack_sizes = [len(group.sample_indices) for group in groups]
    total_tokens = sum(group.total_length for group in groups)
    return {
        "raw_samples": raw_sample_count,
        "packed_items": len(groups),
        "saved_items": raw_sample_count - len(groups),
        "too_long_singletons": too_long_count,
        "multi_sample_packs": sum(1 for size in pack_sizes if size > 1),
        "avg_pack_size": raw_sample_count / len(groups),
        "max_pack_size": max(pack_sizes),
        "avg_pack_fill_ratio": total_tokens / (len(groups) * max_length),
        "max_pack_length": max(group.total_length for group in groups),
    }


def sort_groups_by_first_index(groups: Sequence[PackGroup]) -> list[PackGroup]:
    return sorted(groups, key=lambda group: min(group.sample_indices))


def build_short_task_packing_summary(
    groups_by_task: dict[str, list[PackGroup]],
    raw_counts_by_task: dict[str, int],
    max_length: int,
    too_long_counts_by_task: dict[str, int] | None = None,
) -> dict[str, object]:
    too_long_counts_by_task = too_long_counts_by_task or {}
    task_summaries: dict[str, dict[str, float | int]] = {}
    raw_total = 0
    packed_total = 0
    multi_pack_total = 0

    for task_type, raw_count in sorted(raw_counts_by_task.items()):
        task_groups = groups_by_task.get(task_type, [])
        task_summary = summarize_pack_groups(
            raw_sample_count=raw_count,
            groups=task_groups,
            max_length=max_length,
            too_long_count=int(too_long_counts_by_task.get(task_type, 0)),
        )
        task_summaries[task_type] = task_summary
        raw_total += raw_count
        packed_total += int(task_summary["packed_items"])
        multi_pack_total += int(task_summary["multi_sample_packs"])

    return {
        "raw_samples": raw_total,
        "packed_items": packed_total,
        "saved_items": raw_total - packed_total,
        "multi_sample_packs": multi_pack_total,
        "tasks": task_summaries,
    }
