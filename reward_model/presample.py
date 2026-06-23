"""
presample.py — 训练前离线预采样负样本索引

用途：
  在训练（特别是 grid_search）前，把所有样本的负样本索引一次性采样好，
  存成 .npy 文件。训练时 Dataset 直接用预采样结果，__getitem__ 退化为
  纯 numpy fancy indexing，彻底消除 DataLoader CPU 瓶颈，提升 GPU 利用率。

输出文件（保存在 --output_dir）：
  presample_{num_neg}_{infonce_neg}.npz
    items_order.npy   : (N,)  shot_id 字符串数组（与 embedding 行对应）
    hard_neg_idx.npy  : (N, num_neg)        int32，hard neg 负样本全局索引
    infonce_neg_idx.npy: (N, infonce_neg)   int32，InfoNCE 负样本全局索引

用法示例：
  # 为 grid search 中用到的所有参数组合预采样
  python reward_model/presample.py \\
      --saved_dir  saved \\
      --emb_base   /autodl-tmp \\
      --hard_neg_index reward_model/hard_negatives.json \\
      --output_dir reward_model/presampled \\
      --num_negatives_list   "30,40,50" \\
      --infonce_neg_list     "27,54" \\
  # 单组参数
  python reward_model/presample.py \\
      --num_negatives_list "30" --infonce_neg_list "27"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import _collect_all_shots, ShotItem


# ──────────────────────────────────────────────────────────────
# 核心采样逻辑（与 ShotRewardDataset 保持一致，但批量离线执行）
# ──────────────────────────────────────────────────────────────

class OfflineSampler:
    """离线负样本采样器，逻辑与 ShotRewardDataset 完全对应。"""

    def __init__(
        self,
        items: list[ShotItem],
        hard_neg_index: dict[str, list[str]],
        infonce_window: int,
    ):
        self.items         = items
        self.hard_neg_index = hard_neg_index
        self.infonce_window = infonce_window
        self.rng           = np.random.default_rng()

        N = len(items)
        self._id_to_idx: dict[str, int] = {it.shot_id: i for i, it in enumerate(items)}
        self.id_to_item: dict[str, ShotItem] = {it.shot_id: it for it in items}

        all_idx = np.arange(N, dtype=np.int32)

        # drama → 该剧索引
        self._drama_idx: dict[str, np.ndarray] = {}
        for it in items:
            self._drama_idx.setdefault(it.drama_name, []).append(self._id_to_idx[it.shot_id])
        self._drama_idx = {d: np.array(v, dtype=np.int32) for d, v in self._drama_idx.items()}

        # drama → 其他剧索引（跨剧）
        self._cross_drama_idx: dict[str, np.ndarray] = {}
        for drama, d_idx in self._drama_idx.items():
            mask = np.ones(N, dtype=bool)
            mask[d_idx] = False
            self._cross_drama_idx[drama] = all_idx[mask]

        # "drama|ep" → 该集索引
        self._episode_idx: dict[str, np.ndarray] = {}
        for it in items:
            key = f"{it.drama_name}|{it.episode_idx}"
            self._episode_idx.setdefault(key, []).append(self._id_to_idx[it.shot_id])
        self._episode_idx = {k: np.array(v, dtype=np.int32) for k, v in self._episode_idx.items()}

    # ── 工具 ───────────────────────��──────────────────────────

    def _choice(self, pool: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(pool))
        return self.rng.choice(pool, size=k, replace=False).astype(np.int32)

    # ── Hard Neg ──────────────────────────────────────────────

    def _sample_cross_drama(self, anchor: ShotItem, k: int) -> list[int]:
        pool = self._cross_drama_idx.get(anchor.drama_name, np.arange(len(self.items), dtype=np.int32))
        return self._choice(pool, k).tolist()

    def _sample_intra_drama(self, anchor: ShotItem, k: int) -> list[int]:
        anchor_idx = self._id_to_idx[anchor.shot_id]
        pool = self._drama_idx[anchor.drama_name]
        pool = pool[pool != anchor_idx]
        if len(pool) == 0:
            return self._sample_cross_drama(anchor, k)
        return self._choice(pool, k).tolist()

    def _sample_hard_neg(self, anchor: ShotItem, k: int) -> list[int]:
        hard_ids = self.hard_neg_index.get(anchor.shot_id, [])
        valid_idx = np.array([
            self._id_to_idx[sid] for sid in hard_ids
            if sid in self.id_to_item and sid != anchor.shot_id
        ], dtype=np.int32)
        if len(valid_idx) == 0:
            return self._sample_intra_drama(anchor, k)
        return self._choice(valid_idx, k).tolist()

    def sample_hard_negatives(self, anchor: ShotItem, num_neg: int) -> list[int]:
        """返回 num_neg 个 hard neg 全局索引（去重后补足）。"""
        negs = self._sample_hard_neg(anchor, num_neg)
        seen = {self._id_to_idx[anchor.shot_id]}
        deduped: list[int] = []
        for i in negs:
            if i not in seen:
                seen.add(i); deduped.append(i)
        while len(deduped) < num_neg:
            extra = self._sample_intra_drama(anchor, num_neg - len(deduped))
            for i in extra:
                if i not in seen:
                    seen.add(i); deduped.append(i)
        return deduped[:num_neg]

    # ── InfoNCE Neg ───────────────────────────────────────────

    def _sample_infonce_intra_ep_windowed(self, anchor: ShotItem, k: int) -> list[int]:
        key = f"{anchor.drama_name}|{anchor.episode_idx}"
        ep_idx = self._episode_idx.get(key, np.array([], dtype=np.int32))
        anchor_idx = self._id_to_idx[anchor.shot_id]
        valid = np.array([
            i for i in ep_idx
            if i != anchor_idx
            and abs(self.items[i].shot_index - anchor.shot_index) > self.infonce_window
        ], dtype=np.int32)
        if len(valid) == 0:
            return self._sample_cross_drama(anchor, k)
        return self._choice(valid, k).tolist()

    def _sample_infonce_intra_drama_cross_ep(self, anchor: ShotItem, k: int) -> list[int]:
        drama_idx = self._drama_idx[anchor.drama_name]
        valid = np.array([
            i for i in drama_idx
            if self.items[i].episode_idx != anchor.episode_idx
        ], dtype=np.int32)
        if len(valid) == 0:
            return self._sample_cross_drama(anchor, k)
        return self._choice(valid, k).tolist()

    def sample_infonce_negatives(self, anchor: ShotItem, num_neg: int) -> list[int]:
        """返回 num_neg 个 InfoNCE neg 全局索引（三策略等分，去重补足）。"""
        per = num_neg // 3
        remainder = num_neg - per * 3

        negs: list[int] = []
        negs += self._sample_infonce_intra_ep_windowed(anchor, per)
        negs += self._sample_infonce_intra_drama_cross_ep(anchor, per)
        negs += self._sample_cross_drama(anchor, per + remainder)

        seen = {self._id_to_idx[anchor.shot_id]}
        deduped: list[int] = []
        for i in negs:
            if i not in seen:
                seen.add(i); deduped.append(i)

        while len(deduped) < num_neg:
            extra = self._sample_cross_drama(anchor, num_neg - len(deduped))
            for i in extra:
                if i not in seen:
                    seen.add(i); deduped.append(i)

        return deduped[:num_neg]


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def presample_one(
    items: list[ShotItem],
    hard_neg_index: dict,
    num_neg: int,
    infonce_neg: int,
    infonce_window: int,
    output_dir: Path,
) -> Path:
    """为一组 (num_neg, infonce_neg) 参数预采样，保存 .npz 并返回路径。"""

    out_path = output_dir / f"presample_{num_neg}_{infonce_neg}.npz"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists.")
        return out_path

    N = len(items)
    sampler = OfflineSampler(items, hard_neg_index, infonce_window)

    hard_arr    = np.empty((N, num_neg),    dtype=np.int32)
    infonce_arr = np.empty((N, infonce_neg), dtype=np.int32)

    print(f"  Sampling num_neg={num_neg}, infonce_neg={infonce_neg} for {N} shots ...")
    for i, anchor in enumerate(tqdm(items, ncols=80, leave=False)):
        hard_arr[i]    = sampler.sample_hard_negatives(anchor, num_neg)
        infonce_arr[i] = sampler.sample_infonce_negatives(anchor, infonce_neg)

    # 加载所有 embedding 到内存（float16 节省空间）
    print(f"  Loading embeddings to memory ...")
    D = np.load(str(items[0].text_emb_path)).shape[0]
    text_embs  = np.empty((N, D), dtype=np.float16)
    depth_embs = np.empty((N, D), dtype=np.float16)
    pose_embs  = np.empty((N, D), dtype=np.float16)

    for i, it in enumerate(tqdm(items, ncols=80, leave=False)):
        text_embs[i]  = np.load(str(it.text_emb_path)).astype(np.float16)
        depth_embs[i] = np.load(str(it.depth_emb_path)).astype(np.float16)
        pose_embs[i]  = np.load(str(it.pose_emb_path)).astype(np.float16)

    # 保存：shot_id 顺序、索引、embedding 数据
    shot_ids = np.array([it.shot_id for it in items])
    np.savez_compressed(
        out_path,
        shot_ids        = shot_ids,
        hard_neg_idx    = hard_arr,
        infonce_neg_idx = infonce_arr,
        text_embs       = text_embs,
        depth_embs      = depth_embs,
        pose_embs       = pose_embs,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="预采样负样本索引，加速训练时 DataLoader")
    parser.add_argument("--saved_dir",           type=str, default="saved")
    parser.add_argument("--emb_base",            type=str, default="/root/autodl-tmp")
    parser.add_argument("--hard_neg_index",      type=str, default="reward_model/hard_negatives.json")
    parser.add_argument("--output_dir",          type=str, default="reward_model/presampled")
    parser.add_argument("--num_negatives_list",  type=str, default="40,50",
                        help="逗号分隔的 num_negatives 列表（与 GRID 对应）")
    parser.add_argument("--infonce_neg_list",    type=str, default="27,54",
                        help="逗号分隔的 infonce_num_negatives 列表（与 GRID 对应）")
    parser.add_argument("--infonce_window",      type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析参数列表
    num_neg_list   = [int(x) for x in args.num_negatives_list.split(",")]
    infonce_list   = [int(x) for x in args.infonce_neg_list.split(",")]

    # 加载 hard negative 索引
    hard_neg_index: dict = {}
    if args.hard_neg_index and Path(args.hard_neg_index).exists():
        with open(args.hard_neg_index, encoding="utf-8") as f:
            hard_neg_index = json.load(f)
        print(f"Loaded hard negative index: {len(hard_neg_index)} entries")

    # 扫描所有 shot
    print("Scanning shots ...")
    items = _collect_all_shots(Path(args.saved_dir), Path(args.emb_base))
    print(f"Total valid shots: {len(items)}")

    # 为每组参数组合采样
    combos = [(n, ic) for n in num_neg_list for ic in infonce_list]
    print(f"\nWill presample {len(combos)} parameter combinations:")
    for n, ic in combos:
        print(f"  num_neg={n}, infonce_neg={ic}")

    for n, ic in combos:
        print(f"\n[{n}, {ic}]")
        presample_one(
            items         = items,
            hard_neg_index= hard_neg_index,
            num_neg       = n,
            infonce_neg   = ic,
            infonce_window= args.infonce_window,
            output_dir    = output_dir,
        )

    print(f"\nAll done. Files saved to: {output_dir}")
    print("Next step: run train.py with --presample_dir reward_model/presampled")


if __name__ == "__main__":
    main()
