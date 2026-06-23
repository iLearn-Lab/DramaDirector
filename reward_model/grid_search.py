"""
grid_search.py — 超参数网格搜索

搜索参数：
  --lambda_hard           : hard negative loss 权重
  --val_ratio             : 验证集比例
  --test_ratio            : 测试集比例
  --num_negatives         : hard negative 负样本数
  --infonce_num_negatives : InfoNCE 负样本数（建议为 3 的倍数）

用法：
  python reward_model/grid_search.py
  python reward_model/grid_search.py --dry_run        # 只打印参数组合，不实际运行
  python reward_model/grid_search.py --resume         # 跳过已完成的 run
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ─────────────────────── 搜索空间定义 ────────────────────────

GRID = {
    "lambda_hard": [0.3, 0.5, 1.0],
    "lambda_infonce": [0.3, 0.5, 1.0],
    "lr": [1e-2, 1e-3, 1e-4, 1e-5],
    "val_ratio": [0.05, 0.1],
    "test_ratio": [0.1],
    "num_negatives": [20, 30, 40, 50],
    "infonce_num_negatives": [9, 27, 54],   # 均为 3 的倍数
}

# 固定不变的训练参数（可按需修改）
FIXED_ARGS = {
    "saved_dir":            "saved",
    "emb_base":             "/root/autodl-tmp",
    "hard_neg_index":       "reward_model/hard_negatives.json",
    "hidden_dim":           2048,
    "batch_size":           1024,
    "epochs":               200,
    "lr":                   1e-3,
    "lambda_infonce":       1.0,
    "early_stop_patience":  3,
}

# 目标指标（用于排序和选最优）
METRIC_KEY = "test_recall@10"

# 结果保存目录
RESULTS_ROOT = Path("reward_model/grid_search_results")
SUMMARY_CSV  = RESULTS_ROOT / "summary.csv"


# ─────────────────────── 工具函数 ────────────────────────────

def param_signature(params: dict) -> str:
    """生成参数组合的唯一标识字符串，用于目录命名和去重。"""
    parts = [
        f"lh{params['lambda_hard']}",
        f"vr{params['val_ratio']}",
        f"tr{params['test_ratio']}",
        f"nn{params['num_negatives']}",
        f"in{params['infonce_num_negatives']}",
    ]
    return "_".join(parts)


def build_cmd(params: dict, output_dir: Path) -> list[str]:
    """拼接 train.py 的完整命令行。"""
    cmd = [sys.executable, "reward_model/train.py", "--output_dir", str(output_dir)]
    for k, v in {**FIXED_ARGS, **params}.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def parse_test_metrics(log_text: str) -> dict[str, float]:
    """
    从 train.py 的 stdout 中解析最终 Test Results 行。
    格式示例：
      Test Results | R@1=0.1234 R@5=0.4567 R@10=0.6789 R@20=0.8000 pos_sim=0.5555
    """
    metrics: dict[str, float] = {}
    for line in log_text.splitlines():
        if "Test Results" in line:
            for token in line.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    try:
                        metrics[f"test_{k}"] = float(v)
                    except ValueError:
                        pass
    return metrics


def load_existing_results() -> dict[str, dict]:
    """读取已有的 summary.csv，返回 {signature: row_dict}。"""
    existing: dict[str, dict] = {}
    if SUMMARY_CSV.exists():
        with open(SUMMARY_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["signature"]] = row
    return existing


def append_result_row(row: dict, is_first: bool) -> None:
    """追加一行到 summary.csv。"""
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists() or is_first
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header and SUMMARY_CSV.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────── 主逻辑 ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid search for train.py hyperparameters")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印所有参数组合，不实际运行")
    parser.add_argument("--resume",  action="store_true",
                        help="跳过已完成的 run（基于 signature 去重）")
    parser.add_argument("--timeout", type=int, default=None,
                        help="每个 run 的超时秒数（默认不限制）")
    args = parser.parse_args()

    # 生成所有参数组合
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    all_params = [dict(zip(keys, combo)) for combo in combos]

    total = len(all_params)
    print(f"Total combinations: {total}")
    print(f"Search space:")
    for k, v in GRID.items():
        print(f"  {k}: {v}")
    print()

    if args.dry_run:
        for i, params in enumerate(all_params, 1):
            sig = param_signature(params)
            print(f"[{i:>3}/{total}] {sig}  →  {params}")
        return

    # 加载已完成的 run（resume 模式）
    existing = load_existing_results() if args.resume else {}
    if existing:
        print(f"Resume mode: found {len(existing)} completed runs, will skip them.\n")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for i, params in enumerate(all_params, 1):
        sig = param_signature(params)
        run_dir = RESULTS_ROOT / sig

        print(f"\n{'='*65}")
        print(f"[{i:>3}/{total}]  {sig}")
        print(f"  params: {params}")

        # ── Resume：跳过已完成的 run ──
        if args.resume and sig in existing:
            print(f"  → Skipping (already done)")
            all_results.append(existing[sig])
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(params, run_dir)
        log_path = run_dir / "train.log"

        print(f"  cmd: {' '.join(cmd)}")
        print(f"  log: {log_path}")

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            elapsed = time.time() - t0
            log_text = result.stdout + result.stderr

            # 保存完整日志
            with open(log_path, "w") as f:
                f.write(log_text)

            if result.returncode != 0:
                print(f"  ✗ FAILED (exit {result.returncode}), see {log_path}")
                status = "failed"
                metrics: dict[str, float] = {}
            else:
                metrics = parse_test_metrics(log_text)
                status  = "done"
                print(f"  ✓ Done in {elapsed:.0f}s  |  metrics: {metrics}")

        except subprocess.TimeoutExpired:
            elapsed = args.timeout
            print(f"  ✗ TIMEOUT after {elapsed}s")
            status  = "timeout"
            metrics = {}

        # ── 保存最优模型元信息 ──
        if metrics:
            meta_path = run_dir / "run_meta.json"
            with open(meta_path, "w") as f:
                json.dump({"params": params, "metrics": metrics, "elapsed": elapsed}, f, indent=2)

        # ── 构造结果行 ──
        row: dict = {
            "signature":             sig,
            "status":                status,
            "elapsed_s":             f"{elapsed:.0f}",
            "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{f"param_{k}": v for k, v in params.items()},
            **metrics,
        }
        all_results.append(row)
        append_result_row(row, is_first=(i == 1))

    # ─────────────────────── 汇总排名 ────────────────────────
    print("\n" + "=" * 65)
    print("GRID SEARCH SUMMARY")
    print("=" * 65)

    done_results = [r for r in all_results if r.get("status") == "done" and METRIC_KEY in r]

    if not done_results:
        print("No completed runs found.")
        return

    # 按目标指标降序排列
    done_results.sort(key=lambda r: float(r[METRIC_KEY]), reverse=True)

    print(f"\nTop-10 runs ranked by {METRIC_KEY}:\n")
    header = f"{'Rank':>4}  {'Signature':<55}  {METRIC_KEY:>14}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(done_results[:10], 1):
        print(f"{rank:>4}  {r['signature']:<55}  {float(r[METRIC_KEY]):>14.4f}")

    best = done_results[0]
    print(f"\nBest run: {best['signature']}")
    print(f"  params:")
    for k in GRID.keys():
        print(f"    {k}: {best.get(f'param_{k}', '?')}")
    print(f"  metrics:")
    for k in list(best.keys()):
        if k.startswith("test_") or k.startswith("val_"):
            print(f"    {k}: {best[k]}")
    print(f"\nFull results saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
