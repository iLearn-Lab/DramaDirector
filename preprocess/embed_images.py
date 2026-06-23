from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path
from typing import Generator

import dashscope
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_dashscope_api_key
from project_paths import (
    DEFAULT_DEPTH_EMB_DIR,
    DEFAULT_POSE_EMB_DIR,
    DEFAULT_PROCESSED_SPLIT_DIR,
)


MODEL = "tongyi-embedding-vision-plus"
BATCH_SIZE = 16
MAX_RETRIES = 3
RETRY_DELAY = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def encode_image_base64(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{suffix};base64,{data}"


def call_with_retry(inputs: list[dict], api_key: str) -> list[list[float]]:
    for attempt in range(1, MAX_RETRIES + 1):
        resp = dashscope.MultiModalEmbedding.call(
            api_key=api_key,
            model=MODEL,
            input=inputs,
        )
        if resp.status_code == 200:
            embeddings = resp.output.get("embeddings", [])
            embeddings.sort(key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]
        print(f"  [Attempt {attempt}/{MAX_RETRIES}] API error {resp.status_code}: {resp.message}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {resp.status_code} {resp.message}")


def batch(iterable: list[Path], n: int) -> Generator[list[Path], None, None]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def iter_processed_images(source_root: Path, modality: str) -> Generator[tuple[Path, Path], None, None]:
    """Yield `(image_path, relative_output_path)` pairs for processed_split layout.

    Expected input layout:
      processed_split/<drama>/<episode>/<modality>/keyframe_XXX.png

    Output layout:
      <modality>_emb/<drama>/<episode>/keyframe_XXX.npy
    """
    for image_path in sorted(source_root.glob(f"*/*/{modality}/*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = image_path.relative_to(source_root)
        if len(rel.parts) < 4:
            continue
        drama_name, episode_idx = rel.parts[0], rel.parts[1]
        yield image_path, Path(drama_name) / episode_idx / image_path.with_suffix(".npy").name


def iter_legacy_images(source_dir: Path) -> Generator[tuple[Path, Path], None, None]:
    for image_path in sorted(source_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            rel = image_path.relative_to(source_dir)
            yield image_path, rel.with_suffix(".npy")


def collect_pending(
    source_root: Path,
    modality: str,
    output_dir: Path,
    legacy_flat: bool,
) -> list[tuple[Path, Path]]:
    iterator = iter_legacy_images(source_root / modality) if legacy_flat else iter_processed_images(source_root, modality)
    pending: list[tuple[Path, Path]] = []
    for image_path, rel_out in iterator:
        out_path = output_dir / rel_out
        if out_path.exists():
            continue
        pending.append((image_path, out_path))
    return pending


def process_modality(
    source_root: Path,
    modality: str,
    output_dir: Path,
    api_key: str,
    legacy_flat: bool,
    batch_size: int,
) -> None:
    pending = collect_pending(source_root, modality, output_dir, legacy_flat)
    if not pending:
        print(f"{modality}: nothing to do")
        return

    print(f"{modality}: {len(pending)} images pending -> {output_dir}")
    for chunk in batch(pending, batch_size):
        inputs = []
        valid_pairs: list[tuple[Path, Path]] = []
        for image_path, out_path in chunk:
            try:
                inputs.append({"image": encode_image_base64(image_path)})
                valid_pairs.append((image_path, out_path))
            except Exception as e:
                print(f"  [SKIP] Failed to encode {image_path}: {e}")

        if not inputs:
            continue

        try:
            embeddings = call_with_retry(inputs, api_key)
        except RuntimeError as e:
            print(f"  [ERROR] Batch failed: {e}")
            continue

        for (_, out_path), emb in zip(valid_pairs, embeddings):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, np.array(emb, dtype=np.float32))
            print(f"  Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed depth/pose images using DashScope multimodal embeddings."
    )
    parser.add_argument(
        "--source_root",
        type=str,
        default=str(DEFAULT_PROCESSED_SPLIT_DIR),
        help="Root directory of processed_split assets.",
    )
    parser.add_argument(
        "--depth_output_dir",
        type=str,
        default=str(DEFAULT_DEPTH_EMB_DIR),
        help="Output directory for depth embeddings.",
    )
    parser.add_argument(
        "--pose_output_dir",
        type=str,
        default=str(DEFAULT_POSE_EMB_DIR),
        help="Output directory for pose embeddings.",
    )
    parser.add_argument(
        "--legacy_flat",
        action="store_true",
        help="Use legacy flat input layout: <source_root>/depth and <source_root>/pose.",
    )
    parser.add_argument("--api_key", type=str, default="", help="DashScope API key; optional override for config.py")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or get_dashscope_api_key()
    if not api_key:
        raise ValueError("DashScope API key is required. Please set DASHSCOPE_API_KEY in config.py or pass --api_key.")

    source_root = Path(args.source_root)
    depth_output_dir = Path(args.depth_output_dir)
    pose_output_dir = Path(args.pose_output_dir)

    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    process_modality(
        source_root=source_root,
        modality="depth",
        output_dir=depth_output_dir,
        api_key=api_key,
        legacy_flat=args.legacy_flat,
        batch_size=args.batch_size,
    )
    process_modality(
        source_root=source_root,
        modality="pose",
        output_dir=pose_output_dir,
        api_key=api_key,
        legacy_flat=args.legacy_flat,
        batch_size=args.batch_size,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
