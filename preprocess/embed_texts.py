"""
embed_texts.py — 为 saved/ 下所有 shot 的文本描述生成 tongyi embedding

输出路径：text_emb/{drama_name}/{episode_idx}/keyframe_{shot_index:03d}.npy
与 depth_emb / pose_emb 目录结构完全一致，供 reward_model/dataset.py 使用。

用法：
  cd /root/autodl-tmp
  python embed_texts.py

  # 指定其他路径：
  python embed_texts.py --saved_dir saved --output_dir text_emb
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import dashscope
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_dashscope_api_key
from project_paths import DEFAULT_SAVED_DIR, DEFAULT_TEXT_EMB_DIR

MODEL = "tongyi-embedding-vision-plus"
BATCH_SIZE = 16
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


# ──────────────────────────── API ─────────────────────────────

def call_text_embedding(texts: list[str], api_key: str) -> list[list[float]]:
    """调用 tongyi MultiModalEmbedding 对文本列表取 embedding，带重试。"""
    inputs = [{"text": t} for t in texts]
    for attempt in range(1, MAX_RETRIES + 1):
        resp = dashscope.MultiModalEmbedding.call(
            api_key=api_key,
            model=MODEL,
            input=inputs,
        )
        if resp.status_code == 200:
            embs = sorted(resp.output["embeddings"], key=lambda x: x["index"])
            return [e["embedding"] for e in embs]
        print(f"  [Attempt {attempt}/{MAX_RETRIES}] API error {resp.status_code}: {resp.message}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {resp.status_code} {resp.message}")


def batch(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


# ──────────────────────────── 收集待处理条目 ──────────────────

def collect_pending(saved_dir: Path, output_dir: Path) -> list[dict]:
    """
    扫描 saved_dir 下所有 shot JSON，返回尚未生成 embedding 的条目列表。
    每个条目：{text, out_path}
    """
    pending = []
    for drama_dir in sorted(saved_dir.iterdir()):
        if not drama_dir.is_dir():
            continue
        drama_name = drama_dir.name
        shot_dir = drama_dir / "shot"
        if not shot_dir.exists():
            continue

        for shot_file in sorted(shot_dir.glob("*.json"), key=lambda p: int(p.stem)):
            episode_idx = shot_file.stem
            try:
                with open(shot_file, encoding="utf-8") as f:
                    shots: list[dict] = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {shot_file}: {e}")
                continue

            for shot in shots:
                idx = shot.get("index")
                if idx is None:
                    continue

                out_path = (output_dir / drama_name / episode_idx
                            / f"keyframe_{idx:03d}.npy")
                if out_path.exists():
                    continue  # 已处理，跳过

                # ── 三段式拼接 ──
                segments = []

                # 第一段：镜头构图
                scale  = shot.get("shot_scale", "")
                angle  = shot.get("camera_angle", "")
                motion = shot.get("camera_motion", "")
                if scale or angle or motion:
                    segments.append(f"{scale}{angle}镜头（{motion}）")

                # 第二段：背景/场景
                bg = shot.get("background", "")
                if bg:
                    segments.append(bg)

                # 第三段：人物（去名，去性别）
                for subj in shot.get("subjects", []):
                    position   = subj.get("position", "")
                    clothing   = subj.get("clothing", "")
                    action     = subj.get("action", "")
                    expression = subj.get("expression", "")
                    parts = [f"{position}人物"]
                    if clothing:
                        parts.append(f"身穿{clothing}")
                    if action:
                        parts.append(action)
                    if expression:
                        parts.append(f"表情：{expression}")
                    segments.append("，".join(parts) + "。")

                desc = "；".join(segments)
                if not desc:
                    continue

                pending.append({"text": desc, "out_path": out_path})

    return pending


# ──────────────────────────── 主流程 ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_dir",  type=str, default=str(DEFAULT_SAVED_DIR),
                        help="shot JSON 根目录")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_TEXT_EMB_DIR),
                        help="text embedding 输出根目录")
    parser.add_argument("--api_key",    type=str, default="",
                        help="DashScope API key (optional override; otherwise use config.py)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    api_key = args.api_key or get_dashscope_api_key()
    if not api_key:
        raise ValueError("Missing API key. Please set DASHSCOPE_API_KEY in config.py or pass --api_key.")

    saved_dir  = Path(args.saved_dir)
    output_dir = Path(args.output_dir)

    print(f"Scanning {saved_dir} ...")
    pending = collect_pending(saved_dir, output_dir)
    print(f"Total pending: {len(pending)} shots")

    if not pending:
        print("Nothing to do.")
        return

    done = 0
    errors = 0
    for items in batch(pending, args.batch_size):
        texts     = [it["text"]     for it in items]
        out_paths = [it["out_path"] for it in items]

        try:
            embeddings = call_text_embedding(texts, api_key)
        except RuntimeError as e:
            print(f"  [ERROR] Batch failed: {e}")
            errors += len(items)
            continue

        for out_path, emb in zip(out_paths, embeddings):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), np.array(emb, dtype=np.float32))
            done += 1

        print(f"  Progress: {done}/{len(pending)}  errors={errors}")

    print(f"\nDone. Saved {done} embeddings, {errors} errors.")


if __name__ == "__main__":
    main()
