"""
mine_hard_negatives.py — 离线 Hard Negative 挖掘

策略说明：
  对每个 shot，在同一部剧内找 embedding 相似的 shot 作为 hard negative。
  只在同剧内搜索，避免跨剧负例过于简单。

  相似度过滤：
    计算 anchor 与同剧所有 shot 的余弦相似度，
    取其 90th 百分位数作为动态阈值，只保留相似度 < 阈值的候选。
    这样排除了 top 10% 最相似的 shot（极可能是合法复用镜头或正反打），
    从剩余候选中取相似度最高的 TopK 作为 hard negative。

  与 InfoNCE 负样本的分工：
    InfoNCE 负样本（dataset 在线采样）：同集时间窗口排除 / 同剧跨集 / 跨剧
    Hard Negative（此脚本离线预挖掘）：同剧内 embedding 相似、但经过百分位过滤的困难负例

输出：
  hard_negatives.json：{shot_id: [neg_shot_id_1, ..., neg_shot_id_K]}

用法：
  python mine_hard_negatives.py \
      --saved_dir saved \
      --emb_base  /autodl-tmp \
      --topk      50 \
      --emb_type  depth \
      --output    reward_model/hard_negatives.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# 复用 dataset.py 中的数据扫描逻辑
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import _collect_all_shots, ShotItem


def load_all_embeddings(
    items: list[ShotItem],
    emb_type: str,   # "depth" | "pose" | "mean"
) -> tuple[np.ndarray, list[str]]:
    """
    加载所有 shot 的 embedding，返回 (矩阵, shot_id 列表)。
    emb_type="mean" 时对两路取平均（更鲁棒）。
    """
    vectors = []
    ids = []
    for item in items:
        try:
            if emb_type == "mean":
                d = np.load(str(item.depth_emb_path)).astype(np.float32)
                p = np.load(str(item.pose_emb_path)).astype(np.float32)
                v = (d + p) / 2.0
            else:
                path_map = {
                    "depth": item.depth_emb_path,
                    "pose":  item.pose_emb_path,
                }
                v = np.load(str(path_map[emb_type])).astype(np.float32)
            vectors.append(v)
            ids.append(item.shot_id)
        except Exception as e:
            print(f"[WARN] Skip {item.shot_id}: {e}")

    matrix = np.stack(vectors, axis=0)  # (N, D)
    # L2 归一化
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    matrix = matrix / norms
    return matrix, ids


def mine_hard_negatives_cpu(
    matrix: np.ndarray,   # (N, D) L2-normalized
    ids: list[str],
    topk: int = 50,
    batch_size: int = 512,
    exclude_same_episode: bool = False,
) -> dict[str, list[str]]:
    """
    同剧内挖掘，带 90th 百分位相似度过滤。

    对每个 anchor：
      1. 只考虑同一部剧的候选（排除自身和可选同集）
      2. 计算 anchor 与所有同剧候选的余弦相似度
      3. 以相似度的 90th 百分位数为阈值，只保留相似度 < 阈值的候选
         （排除 top 10% 最相似的，防止正反打/合法复用镜头混入）
      4. 从剩余候选中取相似度最高的 TopK
    """
    N = len(ids)
    matrix_t = torch.from_numpy(matrix)

    # 按剧名建立索引 {drama_name: [global_indices]}
    drama_to_indices: dict[str, list[int]] = {}
    for i, sid in enumerate(ids):
        drama = sid.split("|")[0]
        drama_to_indices.setdefault(drama, []).append(i)

    result: dict[str, list[str]] = {}

    for global_i, anchor_id in enumerate(ids):
        drama = anchor_id.split("|")[0]
        ep    = anchor_id.split("|")[1]

        # 同剧候选池
        candidate_indices = [
            j for j in drama_to_indices.get(drama, [])
            if j != global_i
        ]
        if exclude_same_episode:
            candidate_indices = [
                j for j in candidate_indices
                if ids[j].split("|")[1] != ep
            ]

        if not candidate_indices:
            result[anchor_id] = []
            continue

        cand_tensor = matrix_t[candidate_indices]          # (C, D)
        sims = torch.mv(cand_tensor, matrix_t[global_i])   # (C,)
        sims_np = sims.numpy()

        # 90th 百分位动态阈值
        threshold = float(np.percentile(sims_np, 90))

        # 保留相似度 < 阈值的候选（排除 top 10% 最相似）
        valid_mask = sims_np < threshold
        valid_local = np.where(valid_mask)[0]

        if len(valid_local) == 0:
            result[anchor_id] = []
            continue

        # 从有效候选中取相似度最高的 TopK
        valid_sims = sims_np[valid_local]
        topk_actual = min(topk, len(valid_local))
        top_local = valid_local[np.argsort(valid_sims)[::-1][:topk_actual]]
        result[anchor_id] = [ids[candidate_indices[j]] for j in top_local]

        if global_i % 500 == 0:
            print(f"  [{global_i}/{N}] hard negative mining (CPU)...")

    return result


def mine_hard_negatives_gpu(
    matrix: np.ndarray,
    ids: list[str],
    topk: int = 50,
    batch_size: int = 2048,
    exclude_same_episode: bool = False,
) -> dict[str, list[str]]:
    """
    同剧内挖掘，带 90th 百分位相似度过滤（GPU 版本）。

    对每个 anchor：
      1. 只考虑同一部剧的候选
      2. 以相似度的 90th 百分位数为动态阈值过滤 top 10% 最相似候选
      3. 从剩余候选中取相似度最高的 TopK
    """
    device = torch.device("cuda")
    matrix_t = torch.from_numpy(matrix).to(device)
    N = len(ids)

    drama_to_indices: dict[str, list[int]] = {}
    for i, sid in enumerate(ids):
        drama = sid.split("|")[0]
        drama_to_indices.setdefault(drama, []).append(i)

    result: dict[str, list[str]] = {}

    for global_i, anchor_id in enumerate(ids):
        drama = anchor_id.split("|")[0]
        ep    = anchor_id.split("|")[1]

        candidate_indices = [
            j for j in drama_to_indices.get(drama, [])
            if j != global_i
        ]
        if exclude_same_episode:
            candidate_indices = [
                j for j in candidate_indices
                if ids[j].split("|")[1] != ep
            ]

        if not candidate_indices:
            result[anchor_id] = []
            continue

        cand_idx_t = torch.tensor(candidate_indices, device=device)
        cand_tensor = matrix_t[cand_idx_t]                  # (C, D)
        sims = torch.mv(cand_tensor, matrix_t[global_i])    # (C,)
        sims_cpu = sims.cpu().numpy()

        threshold = float(np.percentile(sims_cpu, 90))

        valid_mask = sims_cpu < threshold
        valid_local = np.where(valid_mask)[0]

        if len(valid_local) == 0:
            result[anchor_id] = []
            continue

        valid_sims = sims_cpu[valid_local]
        topk_actual = min(topk, len(valid_local))
        top_local = valid_local[np.argsort(valid_sims)[::-1][:topk_actual]]
        result[anchor_id] = [ids[candidate_indices[j]] for j in top_local]

        if global_i % 500 == 0:
            print(f"  [{global_i}/{N}] hard negative mining (GPU)...")

    return result


# ─────────────────────── 统计分析 ────────────────────────

def analyze_hard_negatives(
    index: dict[str, list[str]],
    matrix: np.ndarray,
    ids: list[str],
    sample_n: int = 5,
) -> None:
    """打印一些统计信息，帮助验证挖掘质量。"""
    import random
    id_to_row = {sid: i for i, sid in enumerate(ids)}
    matrix_t = torch.from_numpy(matrix)

    print(f"\n[HN Stats] Total anchors: {len(index)}, avg hard negs: "
          f"{sum(len(v) for v in index.values()) / max(len(index), 1):.1f}")

    sample_ids = random.sample(list(index.keys()), min(sample_n, len(index)))
    for sid in sample_ids:
        top1_neg_id = index[sid][0]
        i = id_to_row[sid]
        j = id_to_row[top1_neg_id]
        sim = float(torch.dot(matrix_t[i], matrix_t[j]))
        drama_match = sid.split("|")[0] == top1_neg_id.split("|")[0]
        print(f"  anchor={sid[:40]}... top1_neg_sim={sim:.3f}  same_drama={drama_match}")


# ─────────────────────── CLI ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives for reward model training")
    parser.add_argument("--saved_dir", type=str, default="saved",
                        help="Path to saved/ directory containing drama shot JSONs")
    parser.add_argument("--emb_base",  type=str, default="/root/autodl-tmp",
                        help="Base directory for *_emb folders")
    parser.add_argument("--topk",      type=int, default=50,
                        help="Number of hard negatives to mine per shot")
    parser.add_argument("--emb_type",  type=str, default="mean",
                        choices=["depth", "pose", "mean"],
                        help="Which embedding to use for similarity computation")
    parser.add_argument("--output",    type=str, default="reward_model/hard_negatives.json")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_gpu",   action="store_true",
                        help="Use GPU for faster similarity computation")
    parser.add_argument("--exclude_same_episode", action="store_true",
                        help="Exclude shots from the same episode as hard negatives")
    args = parser.parse_args()

    saved_dir = Path(args.saved_dir)
    emb_base  = Path(args.emb_base)
    output    = Path(args.output)

    print(f"[1/3] Collecting shots from {saved_dir} ...")
    items = _collect_all_shots(saved_dir, emb_base)
    if not items:
        print("No valid shots found. Exiting.")
        return

    print(f"[2/3] Loading embeddings (type={args.emb_type}) ...")
    matrix, ids = load_all_embeddings(items, args.emb_type)
    print(f"      Matrix shape: {matrix.shape}")

    print(f"[3/3] Mining top-{args.topk} hard negatives ...")
    if args.use_gpu and torch.cuda.is_available():
        print("      Using GPU.")
        index = mine_hard_negatives_gpu(
            matrix, ids, args.topk, args.batch_size, args.exclude_same_episode
        )
    else:
        print("      Using CPU.")
        index = mine_hard_negatives_cpu(
            matrix, ids, args.topk, args.batch_size, args.exclude_same_episode
        )

    analyze_hard_negatives(index, matrix, ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=None, separators=(",", ":"))
    print(f"\nSaved hard negative index to {output}  ({len(index)} entries)")


if __name__ == "__main__":
    main()
