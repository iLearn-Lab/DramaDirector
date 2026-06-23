"""
dataset.py — Reward Model 训练数据集

数据结构假设：
  saved/{drama_name}/shot/{episode_idx}.json  → 分镜 JSON 列表
  {base}/{type}_emb/{drama_name}/{episode_idx}/keyframe_{shot_index:03d}.npy

  type 包括：split / depth / pose / text
  （text_emb 由 embed_texts.py 预计算，结构与图像 emb 完全相同）

InfoNCE 负样本策略（三种各占 1/3）：
  1. intra_episode_windowed  : 同剧同集，排除 ±window 时间窗口内的镜头
  2. intra_drama_cross_ep    : 同剧不同集（以集为场景代理，跨集即跨场景）
  3. cross_drama             : 完全不同剧，视觉风格差异最大
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────── 数据项 ────────────────────────

@dataclass
class ShotItem:
    shot_id: str
    drama_name: str
    episode_idx: str
    shot_index: int
    description: str       # description_visual（优先）或旧 description

    text_emb_path:  Path   # tongyi(description_visual) embedding
    depth_emb_path: Path
    pose_emb_path:  Path   # split_emb 不再使用

    dialogue: str  = ""
    emotion:  str  = ""
    duration: float = 0.0


# ─────────────────────── 工具函数 ─────────────────────────

def _emb_path(base: Path, emb_type: str, drama_name: str,
              episode_idx: str, shot_index: int) -> Path:
    """构造 embedding 文件路径。emb_type ∈ {text, depth, pose}

    支持两种目录布局：
      扁平: <base>/{emb_type}_emb/<drama>/<ep>/keyframe_NNN.npy
      双层: <base>/{emb_type}_emb/{emb_type}_emb/<drama>/<ep>/keyframe_NNN.npy
    优先返回存在的路径；若都不存在，返回扁平路径（后续由调用方检查 exists()）。
    """
    flat   = base / f"{emb_type}_emb" / drama_name / episode_idx / f"keyframe_{shot_index:03d}.npy"
    nested = base / f"{emb_type}_emb" / f"{emb_type}_emb" / drama_name / episode_idx / f"keyframe_{shot_index:03d}.npy"
    return nested if nested.exists() else flat


def load_embedding(path: Path) -> torch.Tensor:
    """加载 .npy embedding，返回 float32 Tensor。"""
    arr = np.load(str(path)).astype(np.float32)
    return torch.from_numpy(arr)


def _collect_all_shots(saved_dir: Path, emb_base: Path) -> list[ShotItem]:
    """
    扫描 saved_dir 下所有剧集，构造 ShotItem 列表。
    只收录三种 embedding 都存在的 shot。
    """
    items: list[ShotItem] = []
    skipped = 0

    for drama_dir in sorted(saved_dir.iterdir()):
        if not drama_dir.is_dir():
            continue
        drama_name = drama_dir.name
        shot_dir = drama_dir / "shot"
        if not shot_dir.exists():
            continue

        for shot_file in sorted(shot_dir.glob("*.json"), key=lambda p: int(p.stem)):
            episode_idx = shot_file.stem  # "1", "2", ...
            try:
                with open(shot_file, encoding="utf-8") as f:
                    shots: list[dict] = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {shot_file}: {e}")
                continue

            for shot in shots:
                idx = shot.get("index")
                # 优先 description_visual（无人名的视觉检索文本），回退旧 description
                desc = (shot.get("description_visual") or
                        shot.get("description") or "").strip()
                if idx is None or not desc:
                    skipped += 1
                    continue

                tp = _emb_path(emb_base, "text",  drama_name, episode_idx, idx)
                dp = _emb_path(emb_base, "depth", drama_name, episode_idx, idx)
                pp = _emb_path(emb_base, "pose",  drama_name, episode_idx, idx)

                if not (tp.exists() and dp.exists() and pp.exists()):
                    skipped += 1
                    continue

                items.append(ShotItem(
                    shot_id        = f"{drama_name}|{episode_idx}|{idx}",
                    drama_name     = drama_name,
                    episode_idx    = episode_idx,
                    shot_index     = idx,
                    description    = desc,
                    text_emb_path  = tp,
                    depth_emb_path = dp,
                    pose_emb_path  = pp,
                    dialogue       = (shot.get("dialogue") or "").strip(),
                    emotion        = (shot.get("emotion")  or "").strip(),
                    duration       = float(shot.get("duration") or 0.0),
                ))

    print(f"[Dataset] Loaded {len(items)} valid shots, skipped {skipped}.")
    return items


# ─────────────────────── Dataset ─────────────────────────

class ShotRewardDataset(Dataset):
    """
    每条样本返回：
      anchor_text    : str        当前 shot 描述（query）
      anchor_triplet : (T, T, T)  split/depth/pose embedding
      pos_triplet    : (T, T, T)  ground-truth 同 shot 的 embedding（训练时=anchor）
      neg_triplets   : List[(T,T,T)]  num_negatives 个负样本

    注：anchor_triplet == pos_triplet（自身即正例）；但在 InfoNCE 实现中，
        正例实际上是 in-batch diagonal，这里单独返回方便灵活使用。
    """

    def __init__(
        self,
        saved_dir: Path,
        emb_base: Path,
        num_negatives: int = 4,
        infonce_num_negatives: int = 9,
        infonce_window: int = 5,
        hard_neg_index: dict[str, list[str]] | None = None,
        presample_file: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        saved_dir             : 存放 shot JSON 的根目录（e.g. saved/）
        emb_base              : embedding 根目录（e.g. autodl-tmp/）
        num_negatives         : hard neg loss 每条样本采样的负样本数（从 hard_neg_index 采样）
        infonce_num_negatives : InfoNCE loss 每条样本采样的负样本数（应为 3 的倍数）
        infonce_window        : InfoNCE 同集策略中排除的时间窗口大小（前后各 window 个镜头）
        hard_neg_index        : {shot_id: [hard_neg_shot_id, ...]} 预计算的 hard negative 索引
        presample_file        : presample.py 生成的 .npz 文件路径；
                                指定后 __getitem__ 直接走预采样快路径，跳过在线采样。
        """
        self.num_negatives = num_negatives
        self.infonce_num_negatives = infonce_num_negatives
        self.infonce_window = infonce_window
        self.hard_neg_index = hard_neg_index or {}

        self.items = _collect_all_shots(saved_dir, emb_base)
        if not self.items:
            raise RuntimeError(
                "No valid shots were found for reward-model training. "
                "Check that saved_dir points to shot JSON files and emb_base contains "
                "matching text_emb/depth_emb/pose_emb outputs."
            )
        self.id_to_item: dict[str, ShotItem] = {it.shot_id: it for it in self.items}

        # 分组索引，加速负样本检索
        self._by_drama:   dict[str, list[ShotItem]] = {}   # drama_name → shots
        self._by_episode: dict[str, list[ShotItem]] = {}   # "drama|ep" → shots
        for it in self.items:
            self._by_drama.setdefault(it.drama_name, []).append(it)
            key = f"{it.drama_name}|{it.episode_idx}"
            self._by_episode.setdefault(key, []).append(it)

        self._all_ids = [it.shot_id for it in self.items]

        # ── 预加载所有 embedding 到内存（float16 节省 50% RAM）──
        N = len(self.items)
        D = np.load(str(self.items[0].text_emb_path)).shape[0]
        print(f"[Dataset] Pre-loading embeddings ({N} shots × 3 types × {D}d, float16) ...")
        self._text_embs  = np.empty((N, D), dtype=np.float16)
        self._depth_embs = np.empty((N, D), dtype=np.float16)
        self._pose_embs  = np.empty((N, D), dtype=np.float16)
        self._id_to_idx: dict[str, int] = {}
        for i, it in enumerate(self.items):
            self._text_embs[i]  = np.load(str(it.text_emb_path))
            self._depth_embs[i] = np.load(str(it.depth_emb_path))
            self._pose_embs[i]  = np.load(str(it.pose_emb_path))
            self._id_to_idx[it.shot_id] = i
        mem = (self._text_embs.nbytes + self._depth_embs.nbytes + self._pose_embs.nbytes) / 1024**3
        print(f"[Dataset] Pre-load done. Memory: {mem:.2f} GB")

        # ── 预采样快路径：加载 presample.py 生成的索引和 embedding ──────────────
        self._presample_hard: Optional[np.ndarray] = None    # (N, num_neg)  int32
        self._presample_infonce: Optional[np.ndarray] = None # (N, infonce_neg) int32
        if presample_file is not None and Path(presample_file).exists():
            data = np.load(str(presample_file))
            # 校验 shot_id 顺序与当前 items 一致
            saved_ids = data["shot_ids"].tolist()
            current_ids = [it.shot_id for it in self.items]
            if saved_ids != current_ids:
                print("[Dataset] WARNING: presample shot_id order mismatch, falling back to online sampling.")
            else:
                self._presample_hard   = data["hard_neg_idx"]    # (N, num_neg)
                self._presample_infonce = data["infonce_neg_idx"] # (N, infonce_neg)
                # 替换 memmap embedding 为预采样的内存数组（零磁盘 I/O）
                if "text_embs" in data and "depth_embs" in data and "pose_embs" in data:
                    self._text_embs  = data["text_embs"]   # (N, 1024) float16
                    self._depth_embs = data["depth_embs"]
                    self._pose_embs  = data["pose_embs"]
                    mem = (self._text_embs.nbytes + self._depth_embs.nbytes + self._pose_embs.nbytes) / 1024**3
                    print(f"[Dataset] Loaded presample embeddings from file ({mem:.2f} GB in memory).")
                print(f"[Dataset] Loaded presample file: {presample_file}")
                print(f"[Dataset] __getitem__ will use presampled indices (zero online-sampling overhead).")
        elif presample_file is not None:
            print(f"[Dataset] WARNING: presample_file not found: {presample_file}, falling back to online sampling.")

        # ── 预计算索引数组，消除采样时的 O(N) 遍历 ──
        # （仅在未使用预采样时需要，但保留以兼容在线采样路径）
        all_indices = np.arange(N)
        # drama → 该剧所有 shot 的索引数组
        self._drama_idx: dict[str, np.ndarray] = {
            drama: np.array([self._id_to_idx[it.shot_id] for it in shots])
            for drama, shots in self._by_drama.items()
        }
        # drama → 其他所有剧 shot 的索引数组（跨剧采样用）
        self._cross_drama_idx: dict[str, np.ndarray] = {}
        for drama, d_idx in self._drama_idx.items():
            mask = np.ones(N, dtype=bool)
            mask[d_idx] = False
            self._cross_drama_idx[drama] = all_indices[mask]
        # "drama|ep" → 该集所有 shot 的索引数组
        self._episode_idx: dict[str, np.ndarray] = {
            key: np.array([self._id_to_idx[it.shot_id] for it in shots])
            for key, shots in self._by_episode.items()
        }
        # numpy rng（多进程 worker 中各自独立）
        self._np_rng = np.random.default_rng()

    # ── 负样本采样（全部用预计算索引 + numpy rng，消除 O(N) 遍历）──────────

    def _np_choice(self, idx_arr: np.ndarray, k: int) -> np.ndarray:
        """从索引数组中无放回随机采样 k 个，返回索引。"""
        k = min(k, len(idx_arr))
        return self._np_rng.choice(idx_arr, size=k, replace=False)

    def _sample_random_neg(self, anchor: ShotItem, k: int) -> list[ShotItem]:
        """跨剧随机采样。"""
        pool = self._cross_drama_idx.get(anchor.drama_name, np.arange(len(self.items)))
        return [self.items[i] for i in self._np_choice(pool, k)]

    def _sample_intra_drama_neg(self, anchor: ShotItem, k: int) -> list[ShotItem]:
        """同剧负样本（排除自身）。"""
        anchor_idx = self._id_to_idx[anchor.shot_id]
        pool = self._drama_idx[anchor.drama_name]
        pool = pool[pool != anchor_idx]
        if len(pool) == 0:
            return self._sample_random_neg(anchor, k)
        return [self.items[i] for i in self._np_choice(pool, k)]

    def _sample_hard_embedding_neg(self, anchor: ShotItem, k: int) -> list[ShotItem]:
        """从预计算的 hard negative 索引中采样。"""
        hard_ids = self.hard_neg_index.get(anchor.shot_id, [])
        valid_idx = np.array([
            self._id_to_idx[sid] for sid in hard_ids
            if sid in self.id_to_item and sid != anchor.shot_id
        ], dtype=np.intp)
        if len(valid_idx) == 0:
            return self._sample_intra_drama_neg(anchor, k)
        return [self.items[i] for i in self._np_choice(valid_idx, k)]

    # ── InfoNCE 专用负样本采样 ──────────────────────────────

    def _sample_infonce_intra_episode_windowed(self, anchor: ShotItem, k: int) -> list[ShotItem]:
        """InfoNCE 策略1：同集，排除 anchor 前后 infonce_window 个镜头。"""
        key = f"{anchor.drama_name}|{anchor.episode_idx}"
        ep_idx = self._episode_idx.get(key, np.array([], dtype=np.intp))
        anchor_idx = self._id_to_idx[anchor.shot_id]
        # 排除时间窗口内的镜头
        valid = np.array([
            i for i in ep_idx
            if i != anchor_idx
            and abs(self.items[i].shot_index - anchor.shot_index) > self.infonce_window
        ], dtype=np.intp)
        if len(valid) == 0:
            return self._sample_infonce_cross_drama(anchor, k)
        return [self.items[i] for i in self._np_choice(valid, k)]

    def _sample_infonce_intra_drama_cross_episode(self, anchor: ShotItem, k: int) -> list[ShotItem]:
        """InfoNCE 策略2：同剧不同集。"""
        drama_idx = self._drama_idx[anchor.drama_name]
        valid = np.array([
            i for i in drama_idx
            if self.items[i].episode_idx != anchor.episode_idx
        ], dtype=np.intp)
        if len(valid) == 0:
            return self._sample_infonce_cross_drama(anchor, k)
        return [self.items[i] for i in self._np_choice(valid, k)]

    def _sample_infonce_cross_drama(self, anchor: ShotItem, k: int) -> list[ShotItem]:
        """InfoNCE 策略3：完全跨剧。"""
        pool = self._cross_drama_idx.get(anchor.drama_name, np.arange(len(self.items)))
        return [self.items[i] for i in self._np_choice(pool, k)]

    def _sample_infonce_negatives(self, anchor: ShotItem) -> list[ShotItem]:
        """
        InfoNCE 三策略等分采样，总数 = infonce_num_negatives（应为 3 的倍数）。
        各策略分配 n//3 个，不足时由跨剧策略补全。
        """
        n = self.infonce_num_negatives
        per = n // 3
        remainder = n - per * 3  # 余数分给跨剧（最安全）

        negs: list[ShotItem] = []
        negs += self._sample_infonce_intra_episode_windowed(anchor, per)
        negs += self._sample_infonce_intra_drama_cross_episode(anchor, per)
        negs += self._sample_infonce_cross_drama(anchor, per + remainder)

        # 去重，不足时跨剧补全
        seen = {anchor.shot_id}
        deduped: list[ShotItem] = []
        for it in negs:
            if it.shot_id not in seen:
                seen.add(it.shot_id)
                deduped.append(it)

        while len(deduped) < n:
            extra = self._sample_infonce_cross_drama(anchor, n - len(deduped))
            for it in extra:
                if it.shot_id not in seen:
                    seen.add(it.shot_id)
                    deduped.append(it)

        return deduped[:n]

    def _sample_negatives(self, anchor: ShotItem) -> list[ShotItem]:
        """L_hard_neg 负样本：只从 hard_neg_index 采样。"""
        negs = self._sample_hard_embedding_neg(anchor, self.num_negatives)

        # 去重并补足
        seen = {anchor.shot_id}
        deduped = []
        for it in negs:
            if it.shot_id not in seen:
                seen.add(it.shot_id)
                deduped.append(it)

        # 不足时用同剧负样本补充
        while len(deduped) < self.num_negatives:
            extra = self._sample_intra_drama_neg(anchor, self.num_negatives - len(deduped))
            for it in extra:
                if it.shot_id not in seen:
                    seen.add(it.shot_id)
                    deduped.append(it)

        return deduped[:self.num_negatives]

    # ── Dataset 接口 ────────────────────────────────────────

    def _get_embs(self, shot_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从内存中取 (text, depth, pose) embedding，零磁盘 I/O。保持 float16 减少内存。"""
        i = self._id_to_idx[shot_id]
        return (
            torch.from_numpy(self._text_embs[i]),   # float16
            torch.from_numpy(self._depth_embs[i]),
            torch.from_numpy(self._pose_embs[i]),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        anchor = self.items[idx]
        text_e, depth_e, pose_e = self._get_embs(anchor.shot_id)

        # ── 快路径：使用预采样索引，零 Python 循环开销 ──────────────────
        if self._presample_hard is not None and self._presample_infonce is not None:
            neg_indices         = self._presample_hard[idx]    # (num_neg,) int32
            infonce_neg_indices = self._presample_infonce[idx] # (infonce_neg,) int32
        else:
            # ── 慢路径：在线采样（兜底，与旧逻辑完全一致）──────────────
            negs         = self._sample_negatives(anchor)
            infonce_negs = self._sample_infonce_negatives(anchor)
            neg_indices         = [self._id_to_idx[n.shot_id] for n in negs]
            infonce_neg_indices = [self._id_to_idx[n.shot_id] for n in infonce_negs]

        neg_text  = torch.from_numpy(self._text_embs[neg_indices])   # float16
        neg_depth = torch.from_numpy(self._depth_embs[neg_indices])
        neg_pose  = torch.from_numpy(self._pose_embs[neg_indices])

        infonce_neg_text  = torch.from_numpy(self._text_embs[infonce_neg_indices])   # float16
        infonce_neg_depth = torch.from_numpy(self._depth_embs[infonce_neg_indices])
        infonce_neg_pose  = torch.from_numpy(self._pose_embs[infonce_neg_indices])

        return {
            "text_emb":          text_e,           # (D,)
            "depth_emb":         depth_e,
            "pose_emb":          pose_e,
            "neg_text":          neg_text,          # (num_neg, D)
            "neg_depth":         neg_depth,
            "neg_pose":          neg_pose,
            "infonce_neg_text":  infonce_neg_text,  # (infonce_num_neg, D)
            "infonce_neg_depth": infonce_neg_depth,
            "infonce_neg_pose":  infonce_neg_pose,
            "description":       anchor.description,
            "shot_id":           anchor.shot_id,
        }


# ─────────────────────── collate_fn ──────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """将 Dataset 输出合并为 batch。"""
    return {
        "text_embs":          torch.stack([b["text_emb"]          for b in batch]),   # (B, D)
        "depth_embs":         torch.stack([b["depth_emb"]         for b in batch]),
        "pose_embs":          torch.stack([b["pose_emb"]          for b in batch]),
        "neg_texts":          torch.stack([b["neg_text"]          for b in batch]),   # (B, num_neg, D)
        "neg_depths":         torch.stack([b["neg_depth"]         for b in batch]),
        "neg_poses":          torch.stack([b["neg_pose"]          for b in batch]),
        "infonce_neg_texts":  torch.stack([b["infonce_neg_text"]  for b in batch]),   # (B, infonce_num_neg, D)
        "infonce_neg_depths": torch.stack([b["infonce_neg_depth"] for b in batch]),
        "infonce_neg_poses":  torch.stack([b["infonce_neg_pose"]  for b in batch]),
        "descriptions":       [b["description"] for b in batch],
        "shot_ids":           [b["shot_id"]      for b in batch],
    }
