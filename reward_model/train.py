"""
train.py — Reward Model 训练脚本

损失函数设计：
  L = λ1 * L_infonce + λ2 * L_hard_neg

  1. L_infonce（显式三策略 InfoNCE）：
     - 每个 anchor 使用 dataset 预采样的 infonce_num_negatives 个负样本
     - 负样本来自三策略等分：同集时间窗口排除 / 同剧跨集 / 跨剧
     - 对称损失（text→image + image→text 双向）

  2. L_hard_neg（显式 hard negative ranking loss）：
     - 对每个 anchor，要求 score(anchor, pos) > score(anchor, neg) + margin
     - 使用 dataset 预采样的 num_negatives 个负样本（同剧 embedding 相似度过滤后的 hard neg）
     - 实现为 softmax cross-entropy over [pos, neg1, neg2, ...]

用法：
  python train.py \
      --saved_dir  saved \
      --emb_base   /autodl-tmp \
      --hard_neg_index reward_model/hard_negatives.json \
      --output_dir reward_model/checkpoints \
      --epochs 10 \
      --batch_size 32
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from dataset import ShotRewardDataset, collate_fn
from model import StoryboardRewardModel
from project_paths import DEFAULT_EMB_BASE_DIR, DEFAULT_SAVED_DIR


# ─────────────────────── 网格搜索空间 ───────────────────────

GRID = {
    "lambda_hard":           [3.0, 5.0, 10.0],
    "lambda_infonce":        [3.0, 5.0, 10.0],
    "lr":                    [1e-2],
    "num_negatives":         [50],
    "infonce_num_negatives": [54],   # 建议为 3 的倍数
}

METRIC_KEY = "recall@10"   # 用于排名的目标指标


# ─────────────────────── 损失函数 ────────────────────────

def infonce_loss_symmetric(
    text_vecs:        torch.Tensor,   # (B, H) L2-normalized
    pos_img_vec:      torch.Tensor,   # (B, H) L2-normalized
    infonce_neg_texts:  torch.Tensor, # (B, num_infonce_neg, D)
    infonce_neg_depths: torch.Tensor, # (B, num_infonce_neg, D)
    infonce_neg_poses:  torch.Tensor, # (B, num_infonce_neg, D)
    model:            StoryboardRewardModel,
    temperature:      torch.Tensor,
) -> torch.Tensor:
    """
    显式三策略 InfoNCE（对称）。
    负样本由 dataset 预采样（同集时间窗口排除 / 同剧跨集 / 跨剧各占 1/3）。

    t2i 方向：每个 anchor text 对 [pos_image, neg_image_1..K] 排序，label=0
    i2t 方向：每个 anchor image 对 [pos_text, neg_text_1..K] 排序，label=0
    """
    B, num_neg, D = infonce_neg_depths.shape
    bn = B * num_neg

    # 编码 InfoNCE 负样本图像
    d = model.depth_proj(infonce_neg_depths.view(bn, D))
    p = model.pose_proj(infonce_neg_poses.view(bn, D))
    neg_img_vecs = F.normalize(
        model.image_fuser(torch.stack([d, p], dim=1)), dim=-1
    ).view(B, num_neg, -1)                                  # (B, num_neg, H)

    # 编码 InfoNCE 负样本文本
    neg_text_vecs = F.normalize(
        model.text_proj(infonce_neg_texts.view(bn, D)), dim=-1
    ).view(B, num_neg, -1)                                  # (B, num_neg, H)

    # t2i 方向：anchor text vs [pos_img, neg_imgs...]
    pos_t2i = (text_vecs * pos_img_vec).sum(-1, keepdim=True)       # (B, 1)
    neg_t2i = (text_vecs.unsqueeze(1) * neg_img_vecs).sum(-1)       # (B, num_neg)
    l_t2i = F.cross_entropy(
        torch.cat([pos_t2i, neg_t2i], dim=1) / temperature,
        torch.zeros(B, dtype=torch.long, device=text_vecs.device),
    )

    # i2t 方向：anchor image vs [pos_text, neg_texts...]
    pos_i2t = (pos_img_vec * text_vecs).sum(-1, keepdim=True)       # (B, 1)
    neg_i2t = (pos_img_vec.unsqueeze(1) * neg_text_vecs).sum(-1)    # (B, num_neg)
    l_i2t = F.cross_entropy(
        torch.cat([pos_i2t, neg_i2t], dim=1) / temperature,
        torch.zeros(B, dtype=torch.long, device=text_vecs.device),
    )

    return (l_t2i + l_i2t) / 2


def hard_neg_ranking_loss(
    text_vecs:   torch.Tensor,   # (B, H)
    pos_img_vec: torch.Tensor,   # (B, H)
    neg_texts:   torch.Tensor,   # (B, num_neg, D)  负样本文本 embedding
    neg_depths:  torch.Tensor,   # (B, num_neg, D)
    neg_poses:   torch.Tensor,
    model:       StoryboardRewardModel,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    Hard negative ranking loss（双向）：
      text 侧：anchor text_vec 对 [pos_img, neg_img1..K] 排序
      image 侧：anchor image_vec 对 [pos_text, neg_text1..K] 排序
    """
    B, num_neg, D = neg_depths.shape

    # ── 编码负样本图像 ──
    bn = B * num_neg
    d = model.depth_proj(neg_depths.view(bn, D))
    p = model.pose_proj(neg_poses.view(bn, D))
    neg_img_vecs = F.normalize(
        model.image_fuser(torch.stack([d, p], dim=1)), dim=-1
    ).view(B, num_neg, -1)                               # (B, num_neg, H)

    # ── 编码负样本文本 ──
    neg_text_vecs = F.normalize(
        model.text_proj(neg_texts.view(bn, D)), dim=-1
    ).view(B, num_neg, -1)                               # (B, num_neg, H)

    # ── text → image 方向 ──
    pos_t2i = (text_vecs * pos_img_vec).sum(-1, keepdim=True)          # (B, 1)
    neg_t2i = (text_vecs.unsqueeze(1) * neg_img_vecs).sum(-1)          # (B, num_neg)
    l_t2i   = F.cross_entropy(
        torch.cat([pos_t2i, neg_t2i], dim=1) / temperature,
        torch.zeros(B, dtype=torch.long, device=text_vecs.device),
    )

    # ── image → text 方向 ──
    pos_i2t = (pos_img_vec * text_vecs).sum(-1, keepdim=True)          # (B, 1)
    neg_i2t = (pos_img_vec.unsqueeze(1) * neg_text_vecs).sum(-1)       # (B, num_neg)
    l_i2t   = F.cross_entropy(
        torch.cat([pos_i2t, neg_i2t], dim=1) / temperature,
        torch.zeros(B, dtype=torch.long, device=text_vecs.device),
    )

    return (l_t2i + l_i2t) / 2


# ─────────────────────── 训练循环 ────────────────────────

def train_one_epoch(
    model:       StoryboardRewardModel,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    lambda_infonce: float,
    lambda_hard:    float,
) -> dict[str, float]:
    model.train()
    total_loss = total_infonce = total_hard = 0.0
    n_batches = 0

    for batch in loader:
        text_embs  = batch["text_embs"].to(device, dtype=torch.float32)
        depth_embs = batch["depth_embs"].to(device, dtype=torch.float32)
        pose_embs  = batch["pose_embs"].to(device, dtype=torch.float32)
        neg_texts  = batch["neg_texts"].to(device, dtype=torch.float32)
        neg_depths = batch["neg_depths"].to(device, dtype=torch.float32)
        neg_poses  = batch["neg_poses"].to(device, dtype=torch.float32)
        infonce_neg_texts  = batch["infonce_neg_texts"].to(device, dtype=torch.float32)
        infonce_neg_depths = batch["infonce_neg_depths"].to(device, dtype=torch.float32)
        infonce_neg_poses  = batch["infonce_neg_poses"].to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # ── 编码正样本 ──
        text_vecs         = model.encode_text(text_embs)
        _, image_vecs     = model.encode_images(depth_embs, pose_embs)
        temp              = model.temperature

        # ── 两路 loss ──
        l_infonce = infonce_loss_symmetric(
            text_vecs, image_vecs,
            infonce_neg_texts, infonce_neg_depths, infonce_neg_poses,
            model, temp,
        )
        l_hard = hard_neg_ranking_loss(
            text_vecs, image_vecs,
            neg_texts, neg_depths, neg_poses,
            model, temp,
        )

        loss = lambda_infonce * l_infonce + lambda_hard * l_hard

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        total_infonce += l_infonce.item()
        total_hard    += l_hard.item()
        n_batches += 1

    return {
        "loss":    total_loss    / n_batches,
        "infonce": total_infonce / n_batches,
        "hard_neg": total_hard   / n_batches,
    }


@torch.no_grad()
def evaluate(
    model:  StoryboardRewardModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    评估指标：
      - recall@1/5/10/20：在 batch 内检索正确率
      - mean_pos_score：正例的平均 cosine similarity
    """
    model.eval()
    r1_total = r5_total = r10_total = r20_total = pos_score_total = n_total = 0

    for batch in loader:
        text_embs  = batch["text_embs"].to(device, dtype=torch.float32)
        depth_embs = batch["depth_embs"].to(device, dtype=torch.float32)
        pose_embs  = batch["pose_embs"].to(device, dtype=torch.float32)

        out        = model(text_embs, depth_embs, pose_embs, return_embeddings=True)
        text_vecs  = out["text_vec"]
        image_vecs = out["image_vec"]

        B = text_vecs.size(0)
        sim_matrix = torch.mm(text_vecs, image_vecs.T)  # (B, B)
        labels = torch.arange(B, device=device)

        # Recall@K：排名中正例是否在前 K 位
        ranked = sim_matrix.argsort(dim=1, descending=True)
        for i in range(B):
            rank = (ranked[i] == i).nonzero(as_tuple=True)[0].item()
            if rank < 1:  r1_total += 1
            if rank < 5:  r5_total += 1
            if rank < 10: r10_total += 1
            if rank < 20: r20_total += 1

        pos_score_total += sim_matrix[labels, labels].sum().item()
        n_total += B

    return {
        "recall@1":       r1_total / max(n_total, 1),
        "recall@5":       r5_total / max(n_total, 1),
        "recall@10":      r10_total / max(n_total, 1),
        "recall@20":      r20_total / max(n_total, 1),
        "mean_pos_score": pos_score_total / max(n_total, 1),
    }


@torch.no_grad()
def evaluate_raw(loader: DataLoader, device: torch.device) -> dict[str, float]:
    """
    零样本基线：直接用原始 embedding 计算文本-图像对齐指标，不经过任何投影层。
    image_vec = L2_normalize(mean(depth_emb, pose_emb))
    text_vec  = L2_normalize(text_emb)
    """
    r1 = r5 = r10 = r20 = pos_score = n_total = 0

    for batch in loader:
        text  = F.normalize(batch["text_embs"].to(device, dtype=torch.float32),  dim=-1)
        image = F.normalize(
            (batch["depth_embs"] + batch["pose_embs"]).to(device, dtype=torch.float32) / 2, dim=-1
        )
        B  = text.size(0)
        sim = torch.mm(text, image.T)
        labels = torch.arange(B, device=device)
        ranked = sim.argsort(dim=1, descending=True)
        for i in range(B):
            rank = (ranked[i] == i).nonzero(as_tuple=True)[0].item()
            if rank < 1:  r1  += 1
            if rank < 5:  r5  += 1
            if rank < 10: r10 += 1
            if rank < 20: r20 += 1
        pos_score += sim[labels, labels].sum().item()
        n_total   += B

    return {
        "recall@1":       r1  / max(n_total, 1),
        "recall@5":       r5  / max(n_total, 1),
        "recall@10":      r10 / max(n_total, 1),
        "recall@20":      r20 / max(n_total, 1),
        "mean_pos_score": pos_score / max(n_total, 1),
    }


# ─────────────────────── 主函数 ──────────────────────────

def run_training(args) -> dict[str, float]:
    """执行一次完整训练，返回测试集指标。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 hard negative 索引 ──
    hard_neg_index = {}
    if args.hard_neg_index and Path(args.hard_neg_index).exists():
        with open(args.hard_neg_index, encoding="utf-8") as f:
            hard_neg_index = json.load(f)
        print(f"Loaded hard negative index: {len(hard_neg_index)} entries")

    # ── 数据集 ──
    # 自动查找预采样文件（presample.py 生成的 .npz）
    presample_file = None
    if args.presample_dir:
        fname = f"presample_{args.num_negatives}_{args.infonce_num_negatives}.npz"
        candidate = Path(args.presample_dir) / fname
        if candidate.exists():
            presample_file = candidate
            print(f"[train] Using presample file: {presample_file}")
        else:
            print(f"[train] WARNING: presample file not found ({candidate}), using online sampling.")

    dataset = ShotRewardDataset(
        saved_dir             = Path(args.saved_dir),
        emb_base              = Path(args.emb_base),
        num_negatives         = args.num_negatives,
        infonce_num_negatives = args.infonce_num_negatives,
        infonce_window        = args.infonce_window,
        hard_neg_index        = hard_neg_index,
        presample_file        = presample_file,
    )

    n_test  = max(1, int(len(dataset) * args.test_ratio))
    n_val   = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=8, pin_memory=True,
        prefetch_factor=2, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
        prefetch_factor=2, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
        prefetch_factor=2, persistent_workers=True,
    )

    # ── 自动推断 embedding 维度 ──
    sample_item = dataset.items[0]
    emb_dim = np.load(str(sample_item.text_emb_path)).shape[0]
    print(f"Auto-detected embedding dim: {emb_dim}")

    # ── 模型（全参数可训练，无预训练 encoder，lr 可以大）──
    model = StoryboardRewardModel(
        emb_dim    = emb_dim,
        hidden_dim = args.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params/1e6:.2f}M  (all trainable)")

    # ── 优化器（所有参数同一 lr，无差异化）──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_recall = 0.0
    patience_counter = 0

    # ── 零样本基线（训练前，用原始 embedding 直接评估）──
    print("\n" + "=" * 60)
    print("Pre-training baseline (raw embeddings, no projection):")
    raw_metrics = evaluate_raw(test_loader, device)
    print(
        f"  Raw R@1={raw_metrics['recall@1']:.4f} "
        f"R@5={raw_metrics['recall@5']:.4f} "
        f"R@10={raw_metrics['recall@10']:.4f} "
        f"R@20={raw_metrics['recall@20']:.4f} "
        f"pos_sim={raw_metrics['mean_pos_score']:.4f}"
    )
    print("=" * 60 + "\n")

    for epoch in range(1, args.epochs + 1):
        # ── 训练 ──
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            args.lambda_infonce, args.lambda_hard,
        )
        scheduler.step()

        # ── 验证 ──
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"loss={train_metrics['loss']:.4f} "
            f"(infonce={train_metrics['infonce']:.4f} "
            f"hard={train_metrics['hard_neg']:.4f}) | "
            f"val R@1={val_metrics['recall@1']:.4f} "
            f"R@5={val_metrics['recall@5']:.4f} "
            f"R@10={val_metrics['recall@10']:.4f} "
            f"R@20={val_metrics['recall@20']:.4f} "
            f"pos_sim={val_metrics['mean_pos_score']:.4f} "
            f"temp={model.temperature.item():.4f}"
        )

        # ── 保存最优模型 & Early Stopping（监控 Recall@10）──
        if val_metrics["recall@10"] > best_recall:
            best_recall = val_metrics["recall@10"]
            patience_counter = 0
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  → Saved best model (R@10={best_recall:.4f}) to {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{args.early_stop_patience})")
            if patience_counter >= args.early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    # ── 测试集评估（加载最优 checkpoint）──
    print("\n" + "=" * 60)
    print("Testing on best checkpoint...")
    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test Results | "
        f"R@1={test_metrics['recall@1']:.4f} "
        f"R@5={test_metrics['recall@5']:.4f} "
        f"R@10={test_metrics['recall@10']:.4f} "
        f"R@20={test_metrics['recall@20']:.4f} "
        f"pos_sim={test_metrics['mean_pos_score']:.4f}"
    )
    print("=" * 60)

    # 保存最终模型（含测试结果）
    torch.save({
        "epoch":        best_ckpt["epoch"],
        "model_state":  model.state_dict(),
        "val_metrics":  best_ckpt["val_metrics"],
        "test_metrics": test_metrics,
        "args":         vars(args),
    }, output_dir / "final_model.pt")
    print(f"\nTraining done. Best val Recall@10: {best_recall:.4f}")

    # 显式释放内存（网格搜索多轮运行时防止 OOM）
    del model, optimizer, scheduler
    del train_loader, val_loader, test_loader
    del train_set, val_set, test_set, dataset
    torch.cuda.empty_cache()
    gc.collect()

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_dir",      type=str, default=str(DEFAULT_SAVED_DIR))
    parser.add_argument("--emb_base",       type=str, default=str(DEFAULT_EMB_BASE_DIR))
    parser.add_argument("--hard_neg_index", type=str, default="reward_model/hard_negatives.json")
    parser.add_argument("--output_dir",     type=str, default="reward_model/checkpoints")
    parser.add_argument("--hidden_dim",     type=int, default=2048)
    parser.add_argument("--num_negatives",  type=int, default=30)
    parser.add_argument("--infonce_num_negatives", type=int, default=27,
                        help="InfoNCE 每样本负样本数，建议为 3 的倍数（三策略等分）")
    parser.add_argument("--infonce_window", type=int, default=5,
                        help="InfoNCE 同集策略时间窗口（排除 anchor 前后 N 个镜头）")
    parser.add_argument("--batch_size",     type=int,   default=1024)
    parser.add_argument("--epochs",         type=int,   default=1000)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--lambda_infonce", type=float, default=1.0)
    parser.add_argument("--lambda_hard",    type=float, default=1.0)
    parser.add_argument("--val_ratio",      type=float, default=0.1)
    parser.add_argument("--test_ratio",     type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Early stopping patience: epochs without Recall@10 improvement")
    parser.add_argument("--presample_dir",  type=str,   default=None,
                        help="presample.py 生成的预采样文件所在目录，"
                             "自动按 num_negatives/infonce_num_negatives 匹配文件名。"
                             "指定后 DataLoader worker 零采样开销，显著提升 GPU 利用率。")
    # ── 网格搜索开关 ──
    parser.add_argument("--grid_search",    action="store_true",
                        help="开启网格搜索，遍历 GRID 中定义的所有参数组合")
    parser.add_argument("--grid_resume",    action="store_true",
                        help="跳过已完成的 run（基于 output_dir/grid_summary.csv 去重）")
    args = parser.parse_args()

    if not args.grid_search:
        # ── 普通单次训练 ──
        run_training(args)
        return

    # ── 网格搜索模式 ──
    base_output_dir = Path(args.output_dir)
    summary_csv = base_output_dir / "grid_summary.csv"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # 生成所有参数组合
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    total  = len(combos)
    print(f"\nGrid search: {total} combinations")
    for k, v in GRID.items():
        print(f"  {k}: {v}")

    # 读取已完成的 run（resume 模式）
    done_sigs: set[str] = set()
    if args.grid_resume and summary_csv.exists():
        with open(summary_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "done":
                    done_sigs.add(row["signature"])
        print(f"Resume: {len(done_sigs)} runs already done, skipping.\n")

    all_results: list[dict] = []

    for idx, combo in enumerate(combos, 1):
        grid_params = dict(zip(keys, combo))
        sig = "_".join(f"{k[:2]}{v}" for k, v in grid_params.items())

        print(f"\n{'='*65}")
        print(f"[{idx:>3}/{total}]  {sig}")
        print(f"  {grid_params}")

        if sig in done_sigs:
            print("  → Skipping (already done)")
            continue

        # 覆盖 args 中对应字段
        run_args = argparse.Namespace(**vars(args))
        for k, v in grid_params.items():
            setattr(run_args, k, v)
        run_args.output_dir = str(base_output_dir / sig)

        try:
            test_metrics = run_training(run_args)
            status = "done"
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            test_metrics = {}
            status = "failed"

        row = {
            "signature": sig,
            "status":    status,
            **{f"param_{k}": v for k, v in grid_params.items()},
            **test_metrics,
        }
        all_results.append(row)

        # 追加写入 CSV
        write_header = not summary_csv.exists() or summary_csv.stat().st_size == 0
        with open(summary_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ── 排名汇总 ──
    done = [r for r in all_results if r.get("status") == "done" and METRIC_KEY in r]
    if not done:
        print("\nNo completed runs.")
        return

    done.sort(key=lambda r: float(r[METRIC_KEY]), reverse=True)
    print(f"\n{'='*65}")
    print(f"GRID SEARCH DONE — Top-10 by test {METRIC_KEY}")
    print(f"{'='*65}")
    for rank, r in enumerate(done[:10], 1):
        print(f"  #{rank:>2}  {r['signature']:<55}  {METRIC_KEY}={float(r[METRIC_KEY]):.4f}")

    best = done[0]
    print(f"\nBest: {best['signature']}")
    for k in GRID:
        print(f"  {k}: {best.get(f'param_{k}', '?')}")
    print(f"\nFull results: {summary_csv}")


if __name__ == "__main__":
    main()
