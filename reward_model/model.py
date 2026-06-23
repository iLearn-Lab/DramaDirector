"""
model.py — Reward Model 架构（全 tongyi 预计算，四路对称投影）

数据来源：
  text_emb  : tongyi-embedding-vision-plus 对分镜文本的 embedding（预计算，固定）
  split_emb : tongyi 对原始帧的 embedding（预计算，固定）
  depth_emb : tongyi 对深度图的 embedding（预计算，固定）
  pose_emb  : tongyi 对骨架图的 embedding（预计算，固定）
  → 四路 embedding 维度相同（D=1024），来自同一模型，已在同一 tongyi 语义空间

架构（纯投影 + 融合，无额外 encoder）：
  文本侧：text_emb  → TextProjector(D→H) → L2_norm → text_vec (H)
  图像侧：split/depth/pose_emb
            → ModalityProjector × 3 (D→H)
            → ImageFuser (self-attn + mean-pool)
            → L2_norm → image_vec (H)
  Score  : cosine_sim(text_vec, image_vec)   ← 检索与打分完全等价

优势：
  1. tongyi 已在大规模图文对上预训练，text/image embedding 已有初始对齐
  2. 去掉 BERT，参数量从 ~43M 降至 ~4M，训练速度大幅提升
  3. text_proj 与 split_proj 结构对称，语义空间统一
  4. RL 阶段对生成文本调用 tongyi API 即可得到 text_emb，reward 计算完全一致
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────── 子模块 ───────────────────────────

class ModalityProjector(nn.Module):
    """
    将 tongyi embedding（固定，dim=D）投影到统一隐层维度 H。
    文本侧（TextProjector）和图像三路（split/depth/pose）共享相同结构，
    但参数独立，使各模态可以学到特异性的语义映射。

    结构：Linear(D→H) → LayerNorm → GELU → Dropout → Linear(H→H)
    """

    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., emb_dim) → (..., hidden_dim)
        return self.proj(x)


class ImageFuser(nn.Module):
    """
    将三路图像投影向量（split / depth / pose）融合为单一图像表示。

    self-attention 使三路模态互相感知后 mean-pool，
    让模型学会动态权衡各模态的重要性
    （如动作镜头更关注 pose，场景镜头更关注 depth）。
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm     = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        # image_tokens: (B, 3, H)
        x, _ = self.self_attn(image_tokens, image_tokens, image_tokens)
        x = self.norm(image_tokens + x)      # residual
        x = x.mean(dim=1)                    # mean-pool over 3 modalities → (B, H)
        return self.out_proj(x)


class CrossAttentionReranker(nn.Module):
    """
    可选的离线重排序模块（不用于训练主干，不用于 RL reward）。

    用途：bi-encoder 召回 TopK 后，用 cross-attention 精排，用于离线评估/分析。
    输入：text_vec (B, H) + image_tokens (B, 3, H)
    输出：rerank_score (B,)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm       = nn.LayerNorm(hidden_dim)
        self.norm2      = nn.LayerNorm(hidden_dim)
        self.ffn        = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        text_vec:     torch.Tensor,   # (B, H)
        image_tokens: torch.Tensor,   # (B, 3, H)
    ) -> torch.Tensor:                # (B,)
        q = text_vec.unsqueeze(1)                           # (B, 1, H)
        attended, _ = self.cross_attn(q, image_tokens, image_tokens)
        x = self.norm(q + attended)
        x = self.norm2(x + self.ffn(x))
        return self.score_head(x.squeeze(1)).squeeze(-1)    # (B,)


# ─────────────────────── 主模型 ───────────────────────────

class StoryboardRewardModel(nn.Module):
    """
    全预计算双塔 Reward Model。

    四路对称结构：
      text_emb  → text_proj  ─────────────────────── text_vec (H)
      split_emb → split_proj ─┐
      depth_emb → depth_proj ─┼→ ImageFuser → image_vec (H)
      pose_emb  → pose_proj  ─┘
      score = cosine_sim(text_vec, image_vec)

    所有输入 embedding 均由 tongyi-embedding-vision-plus 预计算，
    模型只学习投影和融合层（~4M 参数）。

    Usage
    -----
    model = StoryboardRewardModel(emb_dim=1024, hidden_dim=512)
    score = model(text_embs, depth_embs, pose_embs)
    # score: (B,) ∈ [-1, 1]
    """

    def __init__(
        self,
        emb_dim:       int   = 1024,
        hidden_dim:    int   = 512,
        num_img_heads: int   = 4,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim

        # ── 三路独立投影器（text / depth / pose，不含 split）──
        self.text_proj  = ModalityProjector(emb_dim, hidden_dim, dropout)
        self.depth_proj = ModalityProjector(emb_dim, hidden_dim, dropout)
        self.pose_proj  = ModalityProjector(emb_dim, hidden_dim, dropout)

        # ── 双路图像融合（depth + pose）──
        self.image_fuser = ImageFuser(hidden_dim, num_img_heads, dropout)

        # ── 温度参数（InfoNCE 用）──
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def encode_text(self, text_embs: torch.Tensor) -> torch.Tensor:
        """
        text_embs: (B, D) 预计算的 tongyi text embedding
        Returns  : (B, H) L2-normalized text_vec
        """
        return F.normalize(self.text_proj(text_embs), dim=-1)

    def encode_images(
        self,
        depth_embs: torch.Tensor,  # (B, D)
        pose_embs:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          image_tokens : (B, 2, H)  depth + pose 两路 token
          image_vec    : (B, H)     L2-normalized，检索 & 打分用
        """
        d = self.depth_proj(depth_embs)               # (B, H)
        p = self.pose_proj(pose_embs)
        image_tokens = torch.stack([d, p], dim=1)     # (B, 2, H)
        image_vec    = self.image_fuser(image_tokens)  # (B, H)
        return image_tokens, F.normalize(image_vec, dim=-1)

    def encode_text_biencoder(
        self,
        texts:  list[str],
        device: torch.device,
        api_key: str = "",
        model_name: str = "tongyi-embedding-vision-plus",
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        RL 推理入口：接受原始文本字符串，调用 tongyi API 获取 text_emb，
        再经 text_proj 投影到 H 维对齐空间。

        与 build_index 中的 image_vec 在同一空间，可以直接 cosine 检索/打分。

        Parameters
        ----------
        texts      : list[str]  生成的 shot 描述文本
        device     : torch.device
        api_key    : DashScope API key；若为空则尝试从环境变量 DASHSCOPE_API_KEY 读取
        model_name : tongyi embedding 模型名
        batch_size : API 调用批量大小（建议 16~32，tongyi 单次限制）

        Returns
        -------
        (B, H) L2-normalized text_vec，与 image_vec 在同一对齐空间
        """
        import dashscope
        from config import get_dashscope_api_key

        _api_key = api_key or get_dashscope_api_key()
        if not _api_key:
            raise ValueError("Missing DashScope API key. Please pass api_key or set DASHSCOPE_API_KEY in config.py.")

        all_embs: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start: start + batch_size]
            inputs = [{"text": t} for t in batch_texts]

            for attempt in range(3):
                resp = dashscope.MultiModalEmbedding.call(
                    api_key    = _api_key,
                    model      = model_name,
                    input      = inputs,
                )
                if resp.status_code == 200:
                    embs = sorted(resp.output["embeddings"], key=lambda x: x["index"])
                    all_embs.extend([e["embedding"] for e in embs])
                    break
                elif attempt < 2:
                    import time; time.sleep(1.0 * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"tongyi API error {resp.status_code}: {resp.message}"
                    )

        text_embs = torch.tensor(all_embs, dtype=torch.float32, device=device)  # (B, D)
        return F.normalize(self.text_proj(text_embs), dim=-1)                   # (B, H)

    def forward(
        self,
        text_embs:  torch.Tensor,  # (B, D)
        depth_embs: torch.Tensor,  # (B, D)
        pose_embs:  torch.Tensor,  # (B, D)
        return_embeddings: bool = False,
    ) -> torch.Tensor | dict:
        """
        Parameters
        ----------
        return_embeddings : True → 返回 dict（训练用）；False → 只返回 scalar score (B,)
        """
        text_vec          = self.encode_text(text_embs)
        _, image_vec      = self.encode_images(depth_embs, pose_embs)
        score             = (text_vec * image_vec).sum(dim=-1)   # (B,)

        if return_embeddings:
            return {
                "text_vec":    text_vec,
                "image_vec":   image_vec,
                "score":       score,
                "temperature": self.temperature,
            }
        return score

    def score_batch(
        self,
        text_embs:  torch.Tensor,
        depth_embs: torch.Tensor,
        pose_embs:  torch.Tensor,
    ) -> torch.Tensor:
        """推理入口：返回 score ∈ [-1, 1]。"""
        with torch.no_grad():
            return self.forward(text_embs, depth_embs, pose_embs)
