"""
dual_head_model.py — Phase 3: 双头 Reward Model

在 Phase 1 模型基础上新增 video_gen_head：
  r_retrieval  : 文本是否能检索到对应图片（继承 Phase 1 的 cosine sim）
  r_video_gen  : 文本是否包含视频生成所需的动态结构信息（镜头、动作、时序）

训练时两路同时优化，推理时通过 α 加权合并：
  R_shot = α·r_retrieval + (1-α)·r_video_gen

α 课程学习建议：RL 初期 α=0.7（偏检索），后期线性衰减至 0.5（均衡）
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 Phase 1 的主体架构
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model import StoryboardRewardModel


# ─────────────────────── 视频生成适配弱标签 ───────────────────────

# 镜头类型词（视频生成必须的）
_LENS_PATTERN = re.compile(
    r"(特写|近景|中景|远景|全景|中近景|大特写|大全景|俯拍|仰拍|平拍|斜角|"
    r"跟拍|推进|拉远|摇镜|移镜|升降镜头)"
)
# 动作/运动词（对 ControlNet / AnimateDiff 重要）
_ACTION_PATTERN = re.compile(
    r"(走向|奔跑|转身|回头|低头|抬头|挥手|点头|摇头|站起|坐下|拥抱|"
    r"握手|推开|扑倒|摔落|颤抖|凝视|微笑|皱眉|哭泣|大笑|呼喊)"
)
# 时序/转场词（视频时序连贯性）
_TEMPORAL_PATTERN = re.compile(
    r"(随即|随后|紧接|转而|此时|片刻后|慢慢|逐渐|突然|瞬间|同时|缓缓)"
)


def _video_gen_weak_label(text: str) -> float:
    """
    规则打弱标签：文本视频生成适配度 ∈ [0, 1]。
    基于镜头词、动作词、时序词的命中数量做 sigmoid 软标签。
    """
    lens_cnt   = len(_LENS_PATTERN.findall(text))
    action_cnt = len(_ACTION_PATTERN.findall(text))
    temporal_cnt = len(_TEMPORAL_PATTERN.findall(text))
    score = lens_cnt * 0.4 + action_cnt * 0.4 + temporal_cnt * 0.2
    # sigmoid：score=2 时约 0.73，score=0 时约 0.5
    return float(1 / (1 + 2.71828 ** -(score - 1.5)))


# ─────────────────────── Video Generation Head ───────────────────────

class VideoGenHead(nn.Module):
    """
    在融合后的文本向量 text_vec (H,) 基础上，预测视频生成适配分 ∈ (0,1)。
    输入：cross-attended 文本向量（已编码了图像结构信息）
    输出：sigmoid score
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text_vec: torch.Tensor) -> torch.Tensor:
        # text_vec: (B, H) → (B,)
        return torch.sigmoid(self.mlp(text_vec).squeeze(-1))


# ─────────────────────── 双头模型 ────────────────────────────────────

class DualHeadRewardModel(StoryboardRewardModel):
    """
    继承 Phase 1 的 StoryboardRewardModel，新增 video_gen_head。

    forward() 额外返回 r_video_gen；
    get_shot_reward() 返回课程学习加权后的单一 scalar。
    """

    def __init__(self, alpha: float = 0.7, **kwargs):
        """
        Parameters
        ----------
        alpha : float
            初始检索权重，RL 训练过程中由外部调度器动态更新。
            R_shot = alpha * r_retrieval + (1-alpha) * r_video_gen
        **kwargs :
            传递给 StoryboardRewardModel 的参数（emb_dim, hidden_dim）
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.video_gen_head = VideoGenHead(kwargs.get("hidden_dim", 512))

    def forward(
        self,
        text_embs:  torch.Tensor,  # (B, D) 预计算文本 embedding
        depth_embs: torch.Tensor,
        pose_embs:  torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | dict:
        # ── 纯双塔编码 ──
        _, image_vec = self.encode_images(depth_embs, pose_embs)
        text_vec     = self.encode_text(text_embs)

        # ── 检索分 ──
        r_retrieval = (text_vec * image_vec).sum(dim=-1)   # (B,)

        # ── 视频生成适配分 ──
        r_video_gen = self.video_gen_head(text_vec)         # (B,) ∈ (0,1)

        # ── 加权合并 ──
        r_shot = self.alpha * r_retrieval + (1 - self.alpha) * r_video_gen

        if return_embeddings:
            return {
                "text_vec":     text_vec,
                "image_vec":    image_vec,
                "r_retrieval":  r_retrieval,
                "r_video_gen":  r_video_gen,
                "r_shot":       r_shot,
                "temperature":  self.temperature,
            }
        return r_shot

    def get_shot_reward(
        self,
        text_embs:  torch.Tensor,
        depth_embs: torch.Tensor,
        pose_embs:  torch.Tensor,
    ) -> torch.Tensor:
        """推理入口：返回加权 R_shot (B,)。"""
        with torch.no_grad():
            return self.forward(text_embs, depth_embs, pose_embs)


# ─────────────────────── 双头训练损失 ────────────────────────────────

def dual_head_loss(
    out:          dict,
    texts:        list[str],
    temperature:  torch.Tensor,
    lambda_retr:  float = 1.0,
    lambda_video: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    联合 loss：
      L_retrieval : InfoNCE（对角线正例，和 Phase 1 train.py 相同）
      L_video     : BCE with weak rule labels

    Returns dict with individual losses for logging.
    """
    import torch.nn.functional as F

    text_vecs  = out["text_vec"]    # (B, H)
    image_vecs = out["image_vec"]   # (B, H)
    r_video    = out["r_video_gen"] # (B,) ∈ (0,1)

    B = text_vecs.size(0)
    device = text_vecs.device

    # ── Retrieval InfoNCE ──
    logits = torch.mm(text_vecs, image_vecs.T) / temperature  # (B, B)
    labels = torch.arange(B, device=device)
    l_retr = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    # ── Video Gen BCE (weak labels) ──
    video_labels = torch.tensor(
        [_video_gen_weak_label(t) for t in texts],
        dtype=torch.float32, device=device,
    )
    l_video = F.binary_cross_entropy(r_video, video_labels)

    total = lambda_retr * l_retr + lambda_video * l_video
    return {
        "total":     total,
        "retrieval": l_retr,
        "video_gen": l_video,
    }


# ─────────────────────── Alpha 课程学习调度 ──────────────────────────

class AlphaScheduler:
    """
    线性衰减 α：RL 初期偏向检索（α_start），末期趋向均衡（α_end）。

    Usage:
        scheduler = AlphaScheduler(model, total_steps=1000)
        for step in range(total_steps):
            scheduler.step()
    """

    def __init__(
        self,
        model:       DualHeadRewardModel,
        total_steps: int,
        alpha_start: float = 0.7,
        alpha_end:   float = 0.5,
    ):
        self.model       = model
        self.total_steps = total_steps
        self.alpha_start = alpha_start
        self.alpha_end   = alpha_end
        self._step       = 0

    def step(self):
        progress = min(self._step / max(self.total_steps, 1), 1.0)
        self.model.alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        self._step += 1

    @property
    def current_alpha(self) -> float:
        return self.model.alpha
