#!/usr/bin/env python3
"""
rl_grpo_train.py — GRPO 强化学习微调分镜描述生成模型

数据集：默认使用裁剪后的 seq_continue-only JSON
       （/root/autodl-tmp/data/verl_storyboard_sft/grpo_seq_continue/{train,val,test}.json），
       专门训练续写能力。
       启动时从 saved/ 建立 plot_summary → (drama, episode) 索引；
       经核验三个 split 的匹配率均 > 99.7%，极少数不匹配的样本直接丢弃（不参与训练）。

奖励公式（每 completion 返回一个标量）：
  R = α·R_retrieval + β·R_video_gen + γ·R_format
  β 从 beta_start 线性衰减至 beta_end，α 从 alpha_start 线性增长至 alpha_end
  （课程学习：初期偏视频生成质量，后期逐步增强检索对齐）

─── 三个奖励维度 ───────────────────────────────────────────────────────────────

R_retrieval ∈ [0, 1]（图像检索维度）
  每个训练 step，LLM 新生成的分镜文本需要实时嵌入，才能与图像做相似度比较。
  流程：
    ① visual_query 字符串（从生成分镜的结构化字段构造，去人名，还原 description_visual 格式）
    ② 实时调用通义 MultiModalEmbedding API → 原始 768-dim embedding
       ★ 这里的 API 调用发生在每个训练 step 的 reward 计算阶段，不是预计算
         （因为 LLM 每步生成的文本都是新的，无法提前嵌入）
         model.py 注释明确写道："RL 阶段对生成文本调用 tongyi API 即可得到 text_emb，
         reward 计算完全一致"。结果有 in-memory 缓存，重复输出不重复调用。
    ③ 用已训练好的、冻结的 StoryboardRewardModel.encode_text() 投影到 2048-dim 对齐空间
    ④ 与启动时用 reward_model.encode_images() 预计算好的 image_vec 做余弦相似度
       （图像侧也使用同一个训练好的 reward model 投影，确保双侧在同一对齐空间）
    ⑤ 取 top-1 相似度（最相似图像），经 (x+1)/2 归一化到 [0,1]

R_video_gen ∈ [0, 1]（视频生成维度）
  LLM Judge（qwen-plus / qwen3.5-plus）对 description_narrative 打分：
  - 动作具体性（能被 ControlNet/AnimateDiff 控制）
  - 视觉丰富度（景别/位置/服装/表情等视觉要素）
  - 时序连贯性（时序转场词，镜头内动态流动）
  - 情绪叙事性（情绪/台词/心理状态融入叙事）
  批量调用 + 缓存；LLM 返回 [0,1] 浮点数，直接使用。

R_format ∈ [-1, 1]（格式合规维度）
  分镜 JSON 必须包含全部 11 个字段（dialogue/speaker/emotion 可为 null，但 key 必须存在）：
    index, shot_scale, camera_angle, camera_motion, subjects, background,
    description_narrative, dialogue, speaker, emotion, duration
  字段存在性（60%）+ 内容字段有效性（30%）+ subjects 结构完整性（10%）

启动命令：
  # 单机双卡（脚本会自动用 torchrun 拉起 2 个 worker）
  python rl_grpo_train.py \\
    --nproc_per_node 2 \\
    --dashscope_api_key YOUR_KEY \\
    --judge_api_key YOUR_KEY \\
    --output_dir rl_grpo_output \\
    --num_train_epochs 2 \\
    --per_device_train_batch_size 2 \\
    --num_generations 4 \\
    --max_completion_length 4096 \\
    --learning_rate 5e-5

  # 单机四卡
  python rl_grpo_train.py \\
    --nproc_per_node 4 \\
    --dashscope_api_key YOUR_KEY \\
    --judge_api_key YOUR_KEY \\
    --output_dir rl_grpo_output \\
    --num_train_epochs 2 \\
    --per_device_train_batch_size 1 \\
    --num_generations 4 \\
    --max_completion_length 4096 \\
    --learning_rate 5e-5

  # 也可以手动 torchrun（脚本内不会重复拉起）
  torchrun --nproc_per_node 4 rl_grpo_train.py \\
    --dashscope_api_key YOUR_KEY \\
    --judge_api_key YOUR_KEY \\
    --output_dir rl_grpo_output \\
    --num_train_epochs 2 \\
    --per_device_train_batch_size 1 \\
    --num_generations 4 \\
    --max_completion_length 4096 \\
    --learning_rate 5e-5

  # 不带 API key 调试（仅 format 奖励，单卡）
  python rl_grpo_train.py
"""

from __future__ import annotations

import os


def _normalize_omp_num_threads_env() -> None:
    """Make libgomp happy before importing torch/unsloth in parent and child workers."""
    raw_value = os.environ.get("OMP_NUM_THREADS", "").strip()
    if not raw_value:
        os.environ["OMP_NUM_THREADS"] = "1"
        return
    try:
        if int(raw_value) > 0:
            return
    except ValueError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"


_normalize_omp_num_threads_env()

# ── Unsloth 必须最先导入（打补丁），否则其 CUDA kernel 优化无法生效 ────────────
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import json
import logging
import re
import sys
import random
import shlex
import subprocess
import threading
import time
import warnings
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as TorchDDP
from datasets import Dataset, load_dataset
from transformers import HfArgumentParser
from trl import GRPOConfig, GRPOTrainer

ROOT_PARENT = Path(__file__).resolve().parent.parent
if str(ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT_PARENT))

from config import get_dashscope_api_key

# ── 分布式环境初始化（torchrun 通过环境变量传递 rank 信息）───────────────────
local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_main    = local_rank == 0
num_gpus   = int(os.environ.get("WORLD_SIZE", 1))

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_llm_device() -> str:
    """Bind each torchrun worker to its own GPU before loading the LLM."""
    if not torch.cuda.is_available():
        return "cpu"
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return "cuda:0"


def patch_ddp_config_access() -> None:
    """Expose `.config` on DDP so Unsloth/TRL can read model config after wrapping."""
    if isinstance(getattr(TorchDDP, "config", None), property):
        return

    def _get_config(self):
        return self.module.config

    def _set_config(self, value):
        self.module.config = value

    TorchDDP.config = property(_get_config, _set_config)


def _bootstrap_local_reward_model() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    reward_model_dir = PROJECT_ROOT / "reward_model"
    reward_model_zip = PROJECT_ROOT / "reward_model.zip"
    if reward_model_dir.exists() or not reward_model_zip.exists():
        return

    # 允许直接从同级 zip 包导入，避免目录尚未解压时启动失败。
    reward_model_zip_str = str(reward_model_zip)
    if reward_model_zip_str not in sys.path:
        sys.path.insert(0, reward_model_zip_str)


_bootstrap_local_reward_model()
try:
    from reward_model.model import StoryboardRewardModel
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "找不到本地 reward_model 包；请确认脚本同级目录下存在 "
        f"`{PROJECT_ROOT / 'reward_model'}` 或 `{PROJECT_ROOT / 'reward_model.zip'}`"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def quote_for_display(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def sync_unsloth_mixed_precision_env(training_args) -> str:
    """同步 Unsloth/Accelerate 的混合精度环境变量，避免默认回落到 fp16。"""
    if getattr(training_args, "bf16", False):
        mp = "bf16"
    elif getattr(training_args, "fp16", False):
        mp = "fp16"
    else:
        mp = "no"

    os.environ["ACCELERATE_MIXED_PRECISION"] = mp
    if hasattr(training_args, "mixed_precision"):
        training_args.mixed_precision = mp
    if hasattr(training_args, "bf16_full_eval"):
        training_args.bf16_full_eval = (mp == "bf16")
    if hasattr(training_args, "fp16_full_eval"):
        training_args.fp16_full_eval = (mp == "fp16")
    return mp


def normalize_vllm_server_base_url(host: str, port: int, base_url: str = "") -> str:
    """统一 vLLM server 地址格式，便于本地/远端机器复用同一份训练脚本。"""
    if base_url:
        url = base_url.strip().rstrip("/")
        if "://" not in url:
            url = "http://" + url
        return url
    return f"http://{host}:{port}"


def resolve_warmup_args(raw_value: float | int) -> tuple[int, float]:
    """Support legacy `warmup_steps=0.05` style by mapping fractional values to warmup_ratio."""
    warmup_value = float(raw_value)
    if warmup_value < 0:
        raise ValueError("warmup_steps / warmup_ratio must be >= 0")
    if warmup_value == 0:
        return 0, 0.0
    if warmup_value < 1:
        return 0, warmup_value
    if warmup_value.is_integer():
        return int(warmup_value), 0.0
    raise ValueError(
        f"warmup_steps={raw_value} 非法：当值 >= 1 时必须是整数；"
        "若想传比例，请使用 0~1 之间的小数。"
    )


def patch_peft_tensor_parallel_compat() -> None:
    """PEFT 0.19.x assumes newer transformers TP symbols that are absent in transformers 4.57.x."""
    try:
        from peft.utils import save_and_load as peft_save_and_load
    except Exception:
        return

    if getattr(peft_save_and_load, "_unsloth_tp_compat_patched", False):
        return

    original = peft_save_and_load._maybe_shard_state_dict_for_tp

    def _wrapped_maybe_shard_state_dict_for_tp(model, state_dict, adapter_name):
        try:
            return original(model, state_dict, adapter_name)
        except ImportError as exc:
            if "EmbeddingParallel" not in str(exc):
                raise
            logger.warning(
                "Skipping PEFT tensor-parallel adapter sharding because current "
                "transformers build lacks EmbeddingParallel; this is safe for this "
                "script's DDP-per-rank loading path."
            )
            return None

    peft_save_and_load._maybe_shard_state_dict_for_tp = _wrapped_maybe_shard_state_dict_for_tp
    peft_save_and_load._unsloth_tp_compat_patched = True


def maybe_relaunch_with_torchrun(args: "RLScriptArgs") -> bool:
    world_size = args.nproc_per_node * args.nnodes
    if world_size <= 1 or "LOCAL_RANK" in os.environ:
        return False

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.nnodes}",
        f"--node_rank={args.node_rank}",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    logger.info("torchrun launch command: %s", quote_for_display(cmd))
    if args.dry_run:
        return True
    subprocess.run(cmd, check=True)
    return True


@functools.lru_cache(None)
def _warning_once_compat(self, msg, *args, **kwargs):
    """兼容 transformers 5.2.0 的 warning_once(category) 调用。

    部分 transformers 代码会写成 logger.warning_once(message, FutureWarning)。
    标准 logging 会把 FutureWarning 当作 %-format 参数，导致
    TypeError: not all arguments converted during string formatting。
    """
    if args and isinstance(args[0], type) and issubclass(args[0], Warning):
        args = args[1:]
    self.warning(msg, *args, **kwargs)


logging.Logger.warning_once = _warning_once_compat
warnings.filterwarnings(
    "ignore",
    message=r"Passing `generation_config` together with generation-related arguments.*",
)


# ══════════════════════════════════════════════════════════════════════════════
# 路径常量
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR       = Path("/root/autodl-tmp")
REWARD_CKPT    = BASE_DIR / "reward_model/checkpoints/la2.0_la2.0_lr0.01_nu50_in54/best_model.pt"
BASE_MODEL_DIR = BASE_DIR / "Qwen/Qwen3-8B"
SFT_CKPT_DIR   = BASE_DIR / "outputs/verl_storyboard_sft/checkpoint-426"
DEPTH_EMB_DIR  = BASE_DIR / "depth_emb"
POSE_EMB_DIR   = BASE_DIR / "pose_emb"
SAVED_DIR      = BASE_DIR / "saved"

SEQ_CONTINUE_RL_DIR = BASE_DIR / "data/verl_storyboard_sft/grpo_seq_continue"
TRAIN_DATA = SEQ_CONTINUE_RL_DIR / "train.json"
VAL_DATA   = SEQ_CONTINUE_RL_DIR / "val.json"
TEST_DATA  = SEQ_CONTINUE_RL_DIR / "test.json"

# 预计算 image_vec 的缓存文件（避免每次启动重新遍历文件夹 + 重跑投影）
IMAGE_VEC_CACHE_FILE = BASE_DIR / "image_vec_cache.pt"

# SFT 用户消息中各节的标记（与 sft_unsloth_multigpu.py 保持一致）
_HISTORY_MARKER  = "\n\n【已生成的分镜描述】\n"
_CONTINUE_SUFFIX = "\n\n请继续生成接下来的分镜描述。"


# ══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RLScriptArgs:
    # ── 多卡启动参数（与 sft_unsloth_multigpu.py 保持一致）────────────────
    nproc_per_node: int = field(default=1, metadata={"help": "单机启动的进程数 / GPU 数"})
    nnodes:         int = field(default=1, metadata={"help": "节点数"})
    node_rank:      int = field(default=0, metadata={"help": "当前节点 rank"})
    master_addr:    str = field(default="127.0.0.1", metadata={"help": "torch.distributed master 地址"})
    master_port:    int = field(default=29500, metadata={"help": "torch.distributed master 端口"})
    dry_run:        bool = field(default=False, metadata={"help": "仅打印 torchrun 启动命令，不真正执行"})

    # ── API Keys ───────────────────────────────────────────────────────────
    dashscope_api_key: str = field(
        default="",
        metadata={"help": "DashScope API key for retrieval embedding and LLM judge; defaults to config.py"},
    )
    judge_api_key: str = field(
        default="",
        metadata={"help": "Optional dedicated LLM judge API key; falls back to dashscope_api_key"},
    )
    tongyi_emb_model: str = field(
        default="tongyi-embedding-vision-flash-2026-03-06",
        metadata={"help": "通义多模态 embedding 模型"},
    )
    judge_model: str = field(
        default="qwen3.5-flash",
        metadata={"help": "LLM Judge 模型，例如 qwen-plus / qwen3.5-plus"},
    )
    emb_api_batch: int = field(default=16, metadata={"help": "Embedding API 批量大小"})
    judge_batch:   int = field(default=4, metadata={"help": "LLM Judge 每次评分的最大 narrative 数"})

    # ── 奖励权重 ───────────────────────────────────────────────────────────
    alpha_start: float = field(default=0.7, metadata={"help": "检索奖励权重 α（固定）"})
    alpha_end:   float = field(default=0.5, metadata={"help": "检索奖励权重 α（固定，与 alpha_start 相同）"})
    beta_start:  float = field(default=0.5, metadata={"help": "视频生成奖励初始权重 β（课程学习：初期偏视频生成）"})
    beta_end:    float = field(default=0.5, metadata={"help": "视频生成奖励最终权重（线性衰减）"})
    gamma:       float = field(default=0.1, metadata={"help": "格式奖励权重 γ"})
    top_k_sim:   int   = field(default=1,   metadata={"help": "检索奖励取 top-1 相似度（最相似图像）"})

    # ── 奖励模型 ───────────────────────────────────────────────────────────
    emb_dim:       int = field(default=768,  metadata={"help": "tongyi embedding 原始维度 D"})
    hidden_dim:    int = field(default=2048, metadata={"help": "reward model 投影维度 H"})
    reward_device: str = field(default="auto", metadata={"help": "reward model 运行设备；'auto' 表示自动绑定到每个 rank 自身的 cuda:{local_rank}"})

    # ── 模型/数据路径 ─────────────────────────────────────────────────────
    base_model_dir: str = field(
        default=str(BASE_MODEL_DIR),
        metadata={"help": "GRPO 基模目录"},
    )
    sft_checkpoint_dir: str = field(
        default=str(SFT_CKPT_DIR),
        metadata={"help": "作为 RL 起点的 LoRA / PEFT checkpoint 目录；留空则只加载基模"},
    )
    train_data_path: str = field(
        default=str(TRAIN_DATA),
        metadata={"help": "GRPO 训练集 JSON（建议使用裁剪后的 seq_continue-only 数据）"},
    )
    val_data_path: str = field(
        default=str(VAL_DATA),
        metadata={"help": "GRPO 验证集 JSON"},
    )
    test_data_path: str = field(
        default=str(TEST_DATA),
        metadata={"help": "GRPO 测试集 JSON（当前脚本默认不参与训练，仅保留做后续评估）"},
    )

    # ── 数据集 ─────────────────────────────────────────────────────────────
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "限制训练样本数（None = 使用全部，建议先用小量调试）"},
    )
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "验证集最大样本数（None = 使用全部）"})
    max_seq_length:   int = field(default=8192)

    # ── 训练超参数（GRPOConfig）────────────────────────────────────────────
    num_train_epochs:            int   = field(default=1,        metadata={"help": "训练轮数"})
    per_device_train_batch_size: int   = field(default=16,        metadata={"help": "每卡训练 batch size"})
    per_device_eval_batch_size:  int   = field(default=4,        metadata={"help": "每卡验证 batch size"})
    gradient_accumulation_steps: int   = field(default=2,        metadata={"help": "梯度累积步数"})
    num_generations:             int   = field(default=4,        metadata={"help": "GRPO 每个 prompt 采样数量"})
    max_completion_length:       int   = field(default=2048,     metadata={"help": "生成时最大 completion token 数"})
    learning_rate:               float = field(default=5e-7,     metadata={"help": "学习率"})
    warmup_steps:                float = field(default=0.05,     metadata={"help": "warmup 设置：0~1 小数按比例解释，>=1 按真实 step 数解释"})
    weight_decay:                float = field(default=0.01,     metadata={"help": "权重衰减"})
    lr_scheduler_type:           str   = field(default="cosine", metadata={"help": "学习率调度类型"})
    logging_steps:               int   = field(default=1,        metadata={"help": "日志记录步数"})
    eval_strategy:               str   = field(default="no",     metadata={"help": "评估策略；设为 no 时训练阶段不做 eval"})
    eval_steps:                  int   = field(default=50,        metadata={"help": "评估间隔步数；仅在 eval_strategy != no 时生效"})
    save_steps:                  int   = field(default=50,       metadata={"help": "保存间隔步数"})
    save_total_limit:            int   = field(default=20,        metadata={"help": "最多保存 checkpoint 数"})
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "生成阶段 batch size；None 表示交给 TRL 自动决定"},
    )

    # ── vLLM 生成后端（可选）─────────────────────────────────────────────
    use_vllm: bool = field(default=False, metadata={"help": "是否使用 vLLM 加速 completion 生成"})
    vllm_mode: str = field(
        default="server",
        metadata={"help": "vLLM 模式：server / colocate；多机迁移建议优先 server"},
    )
    vllm_server_host: str = field(
        default="127.0.0.1",
        metadata={"help": "vLLM server 主机地址；server 模式下生效"},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "vLLM server 端口；server 模式下生效"},
    )
    vllm_server_base_url: str = field(
        default="",
        metadata={"help": "完整 vLLM server URL；留空则自动用 host:port 组装"},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "训练进程等待 vLLM server 的超时时间（秒）"},
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "vLLM 的 tensor parallel 大小；独立单卡 server 时设为 1"},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "vLLM 预留给模型/KV cache 的显存占比；仅 colocate 时一定生效"},
    )

    # ── 其他 ───────────────────────────────────────────────────────────────
    output_dir: str = field(default="/root/autodl-tmp/rl_grpo_output")
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "断点续训的 checkpoint 目录；例如 rl_grpo_output/checkpoint-200"},
    )
    save_completions_log: bool = field(
        default=False,
        metadata={"help": "是否写出 rank0 的 completions_log.jsonl 观察文件"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# SFT 系统提示（与 sft_unsloth_multigpu.py 完全一致）
# ══════════════════════════════════════════════════════════════════════════════
_SHOT_TEMPLATE = """\
[
  {
    "index": 0,
    "shot_scale": "景别（特写/近景/中景/全景/远景）",
    "camera_angle": "拍摄角度（平拍/俯拍/仰拍/侧拍/高角度/低角度）",
    "camera_motion": "镜头运动（固定/推/拉/摇/跟等）",
    "subjects": [
      {
        "name": "角色名",
        "gender": "性别",
        "clothing": "服装描述",
        "position": "在画面中的位置",
        "action": "动作描述",
        "expression": "表情描述"
      }
    ],
    "background": "场景/环境描述",
    "description_narrative": "该镜头内发生了什么（叙事散文）",
    "dialogue": "台词原文，无台词则为 null",
    "speaker": "说话角色名，无则为 null",
    "emotion": "情绪标签，无则为 null",
    "duration": 2.5
  }
]"""

SYSTEM_PROMPT = (
    "你是一名专业的影视分镜脚本编剧助手。\n"
    "根据提供的剧情摘要和角色信息，以及已有的分镜描述，继续生成接下来一批分镜描述。\n\n"
    "【输出格式】\n"
    "严格输出 JSON 数组，每个元素为一个分镜对象，字段如下模板所示：\n"
    f"{_SHOT_TEMPLATE}\n\n"
    "【注意事项】\n"
    "- index 从 0 开始，与已有分镜连续编号\n"
    "- duration 单位为秒，保留一位小数\n"
    "- 除 JSON 数组本身外不要输出任何多余文字\n"
    "- 当本集所有分镜全部输出完毕后，在 JSON 数组末尾的右括号后另起一行输出 <END>"
)


# ══════════════════════════════════════════════════════════════════════════════
# 图像 Embedding 缓存（启动时预计算，避免 reward 时重复推理）
# ══════════════════════════════════════════════════════════════════════════════
# "drama/ep" → (M, H=2048) per-episode 帧矩阵（在 device 上）
_img_vec_cache: Dict[str, torch.Tensor] = {}
# "train"/"val"/"test" → 该 split 所有 ep_key 列表（跨剧负样本备用）
_ep_keys_by_split: Dict[str, List[str]] = {}
# "drama_name" → 该剧所有 ep_key 列表（同剧不同集负样本）
_ep_keys_by_drama: Dict[str, List[str]] = {}


def _load_episode_raw_embs(drama: str, ep: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """读取某 episode 的全部 depth/pose embedding（float32 numpy）。"""
    depth_dir = DEPTH_EMB_DIR / drama / ep
    pose_dir  = POSE_EMB_DIR  / drama / ep
    if not depth_dir.exists() or not pose_dir.exists():
        return None, None

    depth_map = {f.stem: f for f in sorted(depth_dir.glob("keyframe_*.npy"))}
    pose_map  = {f.stem: f for f in sorted(pose_dir.glob("keyframe_*.npy"))}
    stems = sorted(set(depth_map) & set(pose_map))
    if not stems:
        return None, None

    d = np.stack([np.load(str(depth_map[s])).astype(np.float32) for s in stems])
    p = np.stack([np.load(str(pose_map[s])).astype(np.float32)  for s in stems])
    return d, p


def preload_split_image_vecs(
    split_ep_keys: Dict[str, set],
    reward_model:  StoryboardRewardModel,
    device:        str,
):
    """为每个 split 加载 per-episode 帧矩阵，填充 _img_vec_cache 和 _ep_keys_by_split。

    split_ep_keys: {"train": {"drama/ep", ...}, "val": {...}, ...}
    缓存文件按 ep_key 存储，reward model 更新后自动重算。
    """
    ckpt_mtime = REWARD_CKPT.stat().st_mtime if REWARD_CKPT.exists() else 0.0

    # ── 尝试从缓存文件读取 ───────────────────────────────────────────────
    file_cache: Dict[str, torch.Tensor] = {}  # ep_key → CPU tensor
    if IMAGE_VEC_CACHE_FILE.exists():
        try:
            saved = torch.load(IMAGE_VEC_CACHE_FILE, map_location="cpu", weights_only=False)
            meta  = saved.get("meta", {})
            if (meta.get("reward_ckpt")       == str(REWARD_CKPT) and
                meta.get("reward_ckpt_mtime") == ckpt_mtime):
                file_cache = saved.get("episodes", {})
                print(f"[image_vec] 缓存文件有效，已有 {len(file_cache)} 个 episode", flush=True)
            else:
                print("[image_vec] 缓存文件已过期（reward model 已更新），重新计算...", flush=True)
        except Exception as e:
            print(f"[image_vec] 缓存文件读取失败（{e}），重新计算...", flush=True)

    # ── 逐 episode 加载或计算 ────────────────────────────────────────────
    all_needed = set().union(*split_ep_keys.values())
    newly_computed: Dict[str, torch.Tensor] = {}

    for ep_key in sorted(all_needed):
        if ep_key in file_cache:
            _img_vec_cache[ep_key] = file_cache[ep_key].to(device)
        else:
            drama, ep = ep_key.split("/", 1)
            d, p = _load_episode_raw_embs(drama, ep)
            if d is None:
                continue
            d_t = torch.from_numpy(d).float().to(device)
            p_t = torch.from_numpy(p).float().to(device)
            with torch.no_grad():
                _, img_vecs = reward_model.encode_images(d_t, p_t)
            _img_vec_cache[ep_key]    = img_vecs
            newly_computed[ep_key]    = img_vecs.cpu()

    # ── 填充 _ep_keys_by_split 和 _ep_keys_by_drama ─────────────────────
    for split_name, ep_keys in split_ep_keys.items():
        _ep_keys_by_split[split_name] = [k for k in ep_keys if k in _img_vec_cache]
        print(
            f"[image_vec] [{split_name}] {len(_ep_keys_by_split[split_name])} 个 episode "
            f"共 {sum(_img_vec_cache[k].shape[0] for k in _ep_keys_by_split[split_name])} 帧 → {device}",
            flush=True,
        )

    # 按剧名建立索引（跨 split 合并，因为同一部剧可能跨 train/val）
    for ep_key in _img_vec_cache:
        drama = ep_key.split("/", 1)[0]
        _ep_keys_by_drama.setdefault(drama, [])
        if ep_key not in _ep_keys_by_drama[drama]:
            _ep_keys_by_drama[drama].append(ep_key)
    print(
        f"[image_vec] 共 {len(_ep_keys_by_drama)} 部剧，"
        f"每剧集数分布：min={min(len(v) for v in _ep_keys_by_drama.values())} "
        f"max={max(len(v) for v in _ep_keys_by_drama.values())}",
        flush=True,
    )

    # ── 更新缓存文件（仅 rank0）─────────────────────────────────────────
    if is_main and newly_computed:
        merged = {**file_cache, **newly_computed}
        try:
            torch.save(
                {
                    "meta": {
                        "reward_ckpt":       str(REWARD_CKPT),
                        "reward_ckpt_mtime": ckpt_mtime,
                    },
                    "episodes": merged,
                },
                IMAGE_VEC_CACHE_FILE,
            )
            print(f"[image_vec] 缓存已更新 → {IMAGE_VEC_CACHE_FILE}（{len(merged)} 个 episode）", flush=True)
        except Exception as e:
            print(f"[image_vec] 缓存保存失败（{e}）", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 通义 Embedding API（文本侧，带缓存）
# ══════════════════════════════════════════════════════════════════════════════
_raw_emb_cache: Dict[str, np.ndarray] = {}  # text → raw (D,) float32
_dashscope_import_warned = False


def _import_dashscope_or_warn():
    global _dashscope_import_warned
    try:
        import dashscope
        return dashscope
    except ModuleNotFoundError:
        if not _dashscope_import_warned:
            logger.warning(
                "检测到配置了 DashScope API key，但当前环境未安装 `dashscope`；"
                "检索奖励将退化为 0，视频奖励将退化为 0.5。可执行 `python -m pip install dashscope` 修复。"
            )
            _dashscope_import_warned = True
        return None


def get_tongyi_raw_embeddings(
    texts:     List[str],
    api_key:   str,
    model_name:str = "tongyi-embedding-vision-flash-2026-03-06",
    batch_size:int = 16,
) -> np.ndarray:
    """批量获取通义多模态 Embedding（raw D-dim），带 in-memory 缓存和指数退避重试。

    返回 (N, D) float32；若 api_key 为空则返回全零（检索奖励退化为 0）。
    """
    if not api_key:
        dim = next(iter(_raw_emb_cache.values()), np.zeros(768)).shape[-1]
        return np.zeros((len(texts), dim), dtype=np.float32)

    dashscope = _import_dashscope_or_warn()
    if dashscope is None:
        dim = next(iter(_raw_emb_cache.values()), np.zeros(768)).shape[-1]
        return np.zeros((len(texts), dim), dtype=np.float32)

    out = np.zeros((len(texts), 768), dtype=np.float32)
    miss_idx, miss_txt = [], []

    for i, t in enumerate(texts):
        if t in _raw_emb_cache:
            out[i] = _raw_emb_cache[t]
        else:
            miss_idx.append(i)
            miss_txt.append(t)

    for chunk_s in range(0, len(miss_txt), batch_size):
        chunk = miss_txt[chunk_s: chunk_s + batch_size]
        for attempt in range(3):
            try:
                resp = dashscope.MultiModalEmbedding.call(
                    api_key=api_key,
                    model=model_name,
                    input=[{"text": t} for t in chunk],
                )
                if resp.status_code == 200:
                    embs = sorted(resp.output["embeddings"], key=lambda x: x["index"])
                    for j, e in enumerate(embs):
                        emb = np.array(e["embedding"], dtype=np.float32)
                        idx = miss_idx[chunk_s + j]
                        out[idx] = emb
                        _raw_emb_cache[miss_txt[chunk_s + j]] = emb
                    break
                logger.warning(f"Embedding API {resp.status_code}: {resp.message}，重试 {attempt+1}/3")
            except Exception as exc:
                logger.warning(f"Embedding API 异常: {exc}，重试 {attempt+1}/3")
            time.sleep(1.5 ** attempt)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# LLM Judge（视频生成适配度，qwen-plus / qwen3.5-plus）
# ══════════════════════════════════════════════════════════════════════════════
_judge_cache: Dict[str, float] = {}  # completion_key → score
_judge_reason_cache: Dict[str, str] = {}  # completion_key → reasoning
_judge_cache_lock = threading.Lock()  # 并发写缓存时保护

_JUDGE_SEP = "\n§§§\n"  # completion 内各 narrative 之间的缓存 key 分隔符

_JUDGE_SYSTEM = """\
你是一位专业的 AI 短剧视频创作导演，负责评估一组按顺序排列的分镜叙事描述（description_narrative）对 AI 视频生成任务的整体适配度。

每组输入由多条分镜描述组成，代表一个完整的短剧片段。请综合考量整组序列在以下五个维度的表现（各占 20% 权重）：

  1. 动作具体性（20%）：人物动作、姿势、行为指令是否清晰具体，适合 ControlNet / AnimateDiff 等模型进行可控生成。
  2. 视觉丰富度（20%）：景别、空间、表情、服饰、光线等视觉信息是否充足、具象。若视觉元素重复但符合拍摄逻辑（如正反打、同场地延续场景），不视为扣分项。
  3. 分镜连贯性（20%）：分镜衔接是否自然、叙事转换是否顺畅。允许非现实设定或幻想剧情出现，只要内部逻辑自洽无需惩罚。
  4. 情绪叙事性（20%）：人物情绪、心理或台词节奏是否能支撑剧情张力与情感表达（无台词片段不扣分）。
  5. 短剧节奏适配（20%）：整体节奏是否匹配短剧视觉风格（1–5 秒/镜头的节奏感），剧情冲突与推进是否得当。

综合总分 = 五个维度均值，范围 [0, 1]。

输出为 JSON 数组，每组序列对应一个对象：
  - "index"：组别序号（从 1 开始）
  - "score"：综合评分（0~1 浮点数）
  - "reasoning"：不超过 200 字，说明得分高/低的主要原因（可涉及动作清晰度、连贯性、节奏感等）。

示例输出：
[{"index": 1, "score": 0.85, "reasoning": "镜头衔接流畅，情绪递进自然，场景设定清晰"}, 
 {"index": 2, "score": 0.63, "reasoning": "动作描述略笼统，节奏稍显平缓"}]
"""


def llm_judge_completions(
    completion_narratives: List[List[str]],
    api_key:    str,
    model:      str = "qwen3.5-flash",
    batch_size: int = 16,
    max_workers: int = 8,
) -> List[float]:
    """用 LLM 对每个 completion 的完整分镜序列整体打分。

    - 每个 completion 的所有 narratives 作为一组发给 LLM，保留上下文连贯性
    - 并发调用：多个 chunk 同时发出，max_workers 控制并发数（受 API QPS 限制）
    - 结果缓存：相同序列不重复调用（缓存 key = 序列内各 narrative 拼接字符串）
    - 失败即报错：API 异常 / 解析失败时直接抛异常，避免静默退化为中性分

    Returns: 与 completion_narratives 等长的 float list，每个值 ∈ [0, 1]
    """
    if not api_key:
        raise ValueError("judge_api_key 为空，无法调用 LLM Judge。")

    dashscope = _import_dashscope_or_warn()
    if dashscope is None:
        raise ModuleNotFoundError("未安装 `dashscope`，无法调用 LLM Judge。")

    results = [0.5] * len(completion_narratives)

    cache_keys = [_JUDGE_SEP.join(nars) for nars in completion_narratives]

    miss_idx = []
    for i, key in enumerate(cache_keys):
        if key in _judge_cache:
            results[i] = _judge_cache[key]
        else:
            miss_idx.append(i)

    if not miss_idx:
        return results

    # 将 miss_idx 切成若干 chunk，每个 chunk 一次 API call
    chunks = [miss_idx[s: s + batch_size] for s in range(0, len(miss_idx), batch_size)]

    def _call_one_chunk(chunk_idxs: List[int]) -> Dict[int, tuple]:
        """调用一次 API，返回 {completion_index: (score, reasoning)}。"""
        seqs = []
        for seq_num, ci in enumerate(chunk_idxs, 1):
            nars = completion_narratives[ci]
            lines = "\n".join(f"  {j+1}. {n}" for j, n in enumerate(nars))
            seqs.append(f"=== 序列 {seq_num} ===\n{lines}")

        expected_indices = list(range(1, len(chunk_idxs) + 1))
        output_template = ",\n ".join(
            f'{{"index": {idx}, "score": 0.00, "reasoning": "示例理由"}}' for idx in expected_indices
        )
        user_msg = (
            f"请对以下 {len(chunk_idxs)} 组分镜序列整体打分。\n"
            f"你必须返回一个完整 JSON 数组，且只包含这 {len(chunk_idxs)} 个对象；"
            f"index 必须严格且完整地覆盖 {expected_indices}，每个 index 恰好出现一次，不得缺失、不得重复、不得输出额外 index。\n"
            "输出要求：\n"
            '1. 顶层必须是 JSON 数组\n'
            '2. 每个元素格式必须为 {"index": 序号, "score": 0~1浮点数, "reasoning": "不超过200字的评分理由"}\n'
            "3. 不要输出任何 JSON 之外的解释、前后缀、markdown 代码块\n"
            f"完整输出格式示例如下（请严格保留全部 index，只替换 score 和 reasoning 的内容）：\n"
            f"[\n {output_template}\n]\n\n"
            + "\n\n".join(seqs)
        )

        last_error = None

        for attempt in range(3):
            try:
                resp = dashscope.MultiModalConversation.call(
                    api_key=api_key,
                    model=model,
                    messages=[
                        {"role": "system", "content": [{"text": _JUDGE_SYSTEM}]},
                        {"role": "user",   "content": [{"text": user_msg}]},
                    ],
                    enable_thinking=False,
                )
                if resp.status_code == 200:
                    content = resp.output.choices[0].message.content
                    if isinstance(content, list):
                        text = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        ).strip()
                    else:
                        text = str(content).strip()
                    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

                    parsed = None
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        m = re.search(r"\[[\s\S]*?\]", text)
                        if m:
                            try:
                                parsed = json.loads(m.group())
                            except json.JSONDecodeError:
                                pass

                    if isinstance(parsed, list) and len(parsed) > 0:
                        score_map: Dict[int, float] = {}
                        reason_map: Dict[int, str] = {}
                        for item in parsed:
                            if isinstance(item, dict) and "score" in item:
                                idx = int(item.get("index", 0))
                                score_map[idx] = max(0.0, min(1.0, float(item["score"])))
                                reason_map[idx] = str(item.get("reasoning", ""))
                        if score_map:
                            chunk_result = {}
                            for j, ci in enumerate(chunk_idxs):
                                if (j + 1) not in score_map:
                                    raise RuntimeError(
                                        f"Judge 返回缺少 index={j + 1} 的 score：{text[:200]}"
                                    )
                                s = score_map[j + 1]
                                r = reason_map.get(j + 1, "")
                                chunk_result[ci] = (s, r)
                            if reason_map.get(1):
                                print(f"[judge reasoning] {reason_map[1][:150]}", flush=True)
                            return chunk_result
                    last_error = f"Judge 响应无法解析：{text[:200]}"
                else:
                    last_error = f"Judge API {resp.status_code}: {resp.message}"
                    print(f"[judge] {last_error}，重试 {attempt+1}/3", flush=True)
            except Exception as exc:
                last_error = f"Judge 异常: {exc}"
                print(f"[judge] {last_error}，重试 {attempt+1}/3", flush=True)
            time.sleep(2.0 ** attempt)

        fallback_msg = last_error or "Judge API 调用失败"
        print(
            f"[judge] {fallback_msg}，连续 3 次失败，当前 chunk 回退为中性分 0.5",
            flush=True,
        )
        return {ci: (0.5, "") for ci in chunk_idxs}

    # 并发提交所有 chunk
    with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
        future_to_chunk = {executor.submit(_call_one_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(future_to_chunk):
            chunk_result = future.result()
            with _judge_cache_lock:
                for ci, (s, r) in chunk_result.items():
                    results[ci] = s
                    _judge_cache[cache_keys[ci]] = s
                    if r:
                        _judge_reason_cache[cache_keys[ci]] = r

    return results


# ══════════════════════════════════════════════════════════════════════════════
# visual_query 构造（严格还原 description_visual 格式，去人名）
# ══════════════════════════════════════════════════════════════════════════════

def _safe_text_field(value) -> str:
    """将奖励阶段可能遇到的异常字段安全转换为文本。

    生成模型偶尔会把原本应为字符串的字段输出成 dict/list。这里递归提取其中
    的文本片段，避免 reward 阶段因 ``endswith`` 等字符串操作直接崩溃。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = [_safe_text_field(v) for v in value.values()]
        parts = [p for p in parts if p]
        return "，".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = [_safe_text_field(v) for v in value]
        parts = [p for p in parts if p]
        return "，".join(parts)
    return ""


_SHOT_TEXT_FIELDS = (
    "shot_scale",
    "camera_angle",
    "camera_motion",
    "background",
    "description_narrative",
    "dialogue",
    "speaker",
    "emotion",
)
_SUBJECT_TEXT_FIELDS = (
    "name",
    "gender",
    "clothing",
    "position",
    "action",
    "expression",
)


def sanitize_shot_for_reward(shot: dict) -> dict:
    """生成仅供 reward 阶段消费的稳健副本，不修改原始分镜结构。"""
    if not isinstance(shot, dict):
        return {}

    cleaned = dict(shot)
    for field in _SHOT_TEXT_FIELDS:
        if field in cleaned:
            cleaned[field] = _safe_text_field(cleaned.get(field))

    subjects = cleaned.get("subjects", [])
    if isinstance(subjects, dict):
        subjects = [subjects]
    elif not isinstance(subjects, list):
        subjects = []

    normalized_subjects = []
    for subj in subjects:
        if not isinstance(subj, dict):
            continue
        normalized = dict(subj)
        for field in _SUBJECT_TEXT_FIELDS:
            if field in normalized:
                normalized[field] = _safe_text_field(normalized.get(field))
        normalized_subjects.append(normalized)
    cleaned["subjects"] = normalized_subjects
    return cleaned


def construct_visual_query(shot: dict) -> str:
    """从生成分镜的结构化字段重建 description_visual 格式的检索查询。

    格式（与 reward model 训练时 description_visual 一致）：
        "{shot_scale}{camera_angle}镜头（{camera_motion}）；{background}。；
         {position}人物，身穿{clothing}，{action}，表情：{expression}。"

    示例：
        "中景平拍镜头（跟拍）；传统中式建筑走廊，红色柱子。；
         画面右侧人物，身穿淡紫色古装长裙，背对镜头走去，表情：不可见。"

    人名排除，确保与 reward model 训练的 embedding 空间对齐。
    """
    if not isinstance(shot, dict):
        return "空镜头"

    segs: List[str] = []

    # 1. 镜头构图
    scale, angle, motion = (
        _safe_text_field(shot.get("shot_scale", "")),
        _safe_text_field(shot.get("camera_angle", "")),
        _safe_text_field(shot.get("camera_motion", "")),
    )
    cam = f"{scale}{angle}镜头" + (f"（{motion}）" if motion else "")
    segs.append(cam)

    # 2. 背景
    bg = _safe_text_field(shot.get("background", ""))
    if bg:
        segs.append(bg if bg.endswith("。") else bg + "。")

    # 3. 人物（去名，保留所有可视属性）
    subjects = shot.get("subjects", [])
    if isinstance(subjects, dict):
        subjects = [subjects]
    elif not isinstance(subjects, list):
        subjects = []
    for subj in subjects:
        if not isinstance(subj, dict):
            continue
        position   = _safe_text_field(subj.get("position", ""))
        clothing   = _safe_text_field(subj.get("clothing", ""))
        action     = _safe_text_field(subj.get("action", ""))
        expression = _safe_text_field(subj.get("expression", ""))
        parts = [f"{position}人物" if position else "人物"]
        if clothing:
            parts.append(f"身穿{clothing}")
        if action:
            parts.append(action)
        if expression:
            parts.append(f"表情：{expression}")
        segs.append("，".join(parts) + "。")

    return "；".join(segs) if segs else "空镜头"


# ══════════════════════════════════════════════════════════════════════════════
# 各维度奖励计算
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_retrieval_reward(
    reward_model:  StoryboardRewardModel,
    raw_text_embs: np.ndarray,   # (N, D=768)
    ep_key:        str,          # 当前 episode，格式 "drama/ep"
    top_k:         int = 1,
    device:        str = "cpu",
) -> float:
    """对比式检索奖励（控制单一变量）。

    正样本：同剧同集帧
    负样本优先级：
      ① 同剧不同集帧（控制剧集风格/演员/服装变量，只变集数内容）
      ② 跨剧随机集帧（该剧仅一集时降级）
    无负样本时退化为纯正样本奖励。

    R = 0.5 * (r_pos + (1 - r_neg))  ∈ [0, 1]
    """
    same_vecs = _img_vec_cache.get(ep_key)
    if same_vecs is None or raw_text_embs.shape[0] == 0:
        return 0.5

    t = torch.from_numpy(raw_text_embs).float().to(device)
    text_vecs = reward_model.encode_text(t)   # (N, H)

    # 正样本
    k = min(top_k, same_vecs.size(0))
    cos_pos = torch.mm(text_vecs, same_vecs.T).topk(k, dim=1).values.mean().item()
    r_pos   = (cos_pos + 1.0) / 2.0

    # 负样本：① 同剧不同集
    drama = ep_key.split("/", 1)[0]
    same_drama_others = [e for e in _ep_keys_by_drama.get(drama, []) if e != ep_key]
    if same_drama_others:
        neg_key = random.choice(same_drama_others)
    else:
        # ② 降级：跨剧随机集（该剧仅一集）
        all_others = [e for e in _img_vec_cache if e.split("/", 1)[0] != drama]
        neg_key = random.choice(all_others) if all_others else None

    if neg_key is not None:
        neg_vecs = _img_vec_cache[neg_key]
        k2       = min(top_k, neg_vecs.size(0))
        cos_neg  = torch.mm(text_vecs, neg_vecs.T).topk(k2, dim=1).values.mean().item()
        r_neg    = (cos_neg + 1.0) / 2.0
        return 0.5 * (r_pos + (1.0 - r_neg))
    return r_pos


# 分镜中所有必须存在的字段（key 必须在 JSON 中出现，值可为 null）
_ALL_SHOT_FIELDS = frozenset({
    "index", "shot_scale", "camera_angle", "camera_motion",
    "subjects", "background", "description_narrative",
    "dialogue", "speaker", "emotion", "duration",
})
# 内容字段：不仅 key 要存在，值也不能为空/null
_CONTENT_FIELDS = frozenset({
    "shot_scale", "camera_angle", "camera_motion",
    "subjects", "background", "description_narrative",
})
# subjects 内每个人物的字段
_SUBJECT_FIELDS = frozenset({"name", "gender", "clothing", "position", "action", "expression"})


def compute_format_reward(shots: List[dict]) -> float:
    """格式合规奖励。

    分镜 JSON 应包含全部 11 个字段（dialogue/speaker/emotion 允许 null 值，但 key 必须存在）。
    - 字段存在性（60%）：11 个 key 全部出现在 JSON 中
    - 内容字段有效性（30%）：6 个内容字段的值非空、非 null
    - subjects 结构完整性（10%）：subjects 列表中每个人物的 6 个子字段覆盖率

    返回 [0, 1]
    """
    if not shots:
        return 0.0

    per_shot = []
    for shot in shots:
        # 字段存在性（key 是否出现，不管值是否为 null）
        presence = sum(1 for f in _ALL_SHOT_FIELDS if f in shot) / len(_ALL_SHOT_FIELDS)

        # 内容字段有效性（值必须非 null、非空字符串、非空列表）
        def _valid(v) -> bool:
            if v is None:
                return False
            if isinstance(v, str):
                return bool(v.strip())
            if isinstance(v, (list, dict)):
                return bool(v)
            return True

        content_ok = sum(1 for f in _CONTENT_FIELDS if _valid(shot.get(f))) / len(_CONTENT_FIELDS)

        # subjects 人物结构完整性
        subjects = shot.get("subjects")
        if isinstance(subjects, list) and subjects:
            valid_subjects = [s for s in subjects if isinstance(s, dict)]
            if not valid_subjects:
                subj_score = 0.0
            else:
                subj_score = sum(
                    sum(1 for sf in _SUBJECT_FIELDS if _valid(s.get(sf))) / len(_SUBJECT_FIELDS)
                    for s in valid_subjects
                ) / len(valid_subjects)
        else:
            subj_score = 0.0

        per_shot.append(0.60 * presence + 0.30 * content_ok + 0.10 * subj_score)

    return float(np.mean(per_shot))


def parse_shots(completion: str) -> List[dict]:
    """从模型 completion 中解析分镜 JSON 数组（容错）。"""
    # 剥离 thinking 内容（兜底，正常情况下 enable_thinking=False 已从源头禁用）
    text = re.sub(r"<think>[\s\S]*?</think>", "", completion)
    text = text.replace("<END>", "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict)]
    except json.JSONDecodeError:
        pass
    # 正则兜底：最长 [...] 块
    for m in re.finditer(r"\[[\s\S]*?\]", text):
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [s for s in parsed if isinstance(s, dict)]
        except json.JSONDecodeError:
            continue
    return []


def normalize_completion_to_complete_json_array(completion: str) -> Tuple[str, bool]:
    """将 completion 修剪为“只保留完整分镜对象”的 JSON 数组。

    目标：
    - 若 completion 本身已是完整 JSON 数组，则原样规范化返回
    - 若因为 hitting max_completion_length 导致最后一个分镜对象被截断，
      则丢弃最后这个不完整对象，只保留前面完整的分镜

    返回：
    - normalized_text: 规范化后的 JSON 数组文本；若完全无法恢复，则返回清洗后的原文本
    - repaired: 是否发生了“截断到上一个完整分镜”的修复
    """
    text = re.sub(r"<think>[\s\S]*?</think>", "", completion)
    text = text.replace("<END>", "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            shots = [s for s in parsed if isinstance(s, dict)]
            return json.dumps(shots, ensure_ascii=False, indent=2), False
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    if start < 0:
        return text, False

    decoder = json.JSONDecoder()
    cursor = start + 1
    shots: List[dict] = []

    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        if text[cursor] == "]":
            return json.dumps(shots, ensure_ascii=False, indent=2), False
        if text[cursor] == ",":
            cursor += 1
            continue
        try:
            item, next_cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            shots.append(item)
        cursor = next_cursor

    if shots:
        return json.dumps(shots, ensure_ascii=False, indent=2), True
    return text, False


# ══════════════════════════════════════════════════════════════════════════════
# RewardOrchestrator — 组合三路奖励
# ══════════════════════════════════════════════════════════════════════════════

class RewardOrchestrator:
    """GRPOTrainer 的 reward_funcs 参数。

    签名（TRL 约定）：
        __call__(completions: List[str], **kwargs) → List[float]
    **kwargs 包含 Dataset 中除 "prompt" 外的所有列：drama_name、episode_idx。

    Unsloth GRPOTrainer 会读取 reward_funcs[i].__name__，类实例需显式声明。
    """
    __name__ = "reward_orchestrator"

    def __init__(
        self,
        args:         RLScriptArgs,
        reward_model: StoryboardRewardModel,
        total_steps:  int = 1000,
    ):
        self.args         = args
        self.reward_model = reward_model
        self.total_steps  = total_steps
        self._step        = 0
        # judge_api_key 优先用 judge_api_key，fallback 到 dashscope_api_key
        self._judge_key = args.judge_api_key or args.dashscope_api_key

    @property
    def alpha(self) -> float:
        """α 固定（alpha_start == alpha_end 时退化为常数）。"""
        p = min(self._step / max(self.total_steps, 1), 1.0)
        return self.args.alpha_start + (self.args.alpha_end - self.args.alpha_start) * p

    @property
    def beta(self) -> float:
        """β 随训练步数从 beta_start 线性衰减到 beta_end（课程学习：初期偏视频生成）。"""
        p = min(self._step / max(self.total_steps, 1), 1.0)
        return self.args.beta_start + (self.args.beta_end - self.args.beta_start) * p

    def __call__(
        self,
        completions: List[str],
        drama_name:  List[str] = None,
        episode_idx: List[str] = None,
        split:       List[str] = None,
        **kwargs,
    ) -> List[float]:
        self._step += 1
        a, b, g = self.alpha, self.beta, self.args.gamma
        split_name = split[0] if split else "train"
        _t = time.perf_counter

        # ── 1. 解析所有 completion 的分镜 ───────────────────────────────
        t0 = _t()
        normalized_completions: List[str] = []
        repaired_completions = 0
        for completion in completions:
            normalized_text, repaired = normalize_completion_to_complete_json_array(completion)
            normalized_completions.append(normalized_text)
            repaired_completions += int(repaired)

        all_shots_list = [parse_shots(c) for c in normalized_completions]
        safe_shots_list = [
            [sanitize_shot_for_reward(shot) for shot in shots]
            for shots in all_shots_list
        ]
        t_parse = _t() - t0

        # ── 2. 收集所有 visual_query（批量 Embedding API）────────────────
        all_queries:  List[str]            = []
        query_slices: List[Tuple[int,int]] = []
        for shots in safe_shots_list:
            s = len(all_queries)
            all_queries.extend(construct_visual_query(sh) for sh in shots)
            query_slices.append((s, len(all_queries)))

        t0 = _t()
        if all_queries and self.args.dashscope_api_key:
            all_raw_embs = get_tongyi_raw_embeddings(
                all_queries,
                self.args.dashscope_api_key,
                self.args.tongyi_emb_model,
                self.args.emb_api_batch,
            )
        else:
            all_raw_embs = np.zeros((max(len(all_queries), 1), self.args.emb_dim), dtype=np.float32)
        t_emb = _t() - t0

        # ── 3. 按 completion 整体发给 LLM Judge（保留分镜间上下文）─────────
        # 每个 completion 的全部 narratives 作为一组，judge 看到完整序列才能评估连贯性
        completion_narratives: List[List[str]] = [
            [_safe_text_field(sh.get("description_narrative", "")) for sh in shots]
            for shots in safe_shots_list
        ]

        t0 = _t()
        if any(completion_narratives) and self._judge_key:
            completion_video_scores = llm_judge_completions(
                completion_narratives,
                self._judge_key,
                self.args.judge_model,
                self.args.judge_batch,
            )
        else:
            completion_video_scores = [0.5] * len(all_shots_list)
        t_judge = _t() - t0

        # ── 4. 逐 completion 合并三路奖励 ────────────────────────────────
        t0 = _t()
        rewards: List[float] = []
        _is_rank0 = (os.environ.get("LOCAL_RANK", "0") == "0")
        should_log_completions = _is_rank0 and self.args.save_completions_log
        log_entries: List[dict] = [] if should_log_completions else None

        for i, shots in enumerate(all_shots_list):
            q_start, q_end = query_slices[i]

            r_format = 2.0 * compute_format_reward(shots) - 1.0

            if not shots:
                rewards.append(-1.0)
                if should_log_completions:
                    log_entries.append({
                        "ep_key":      f"{drama_name[i]}/{episode_idx[i]}" if drama_name and episode_idx else "",
                        "reward":      -1.0,
                        "r_retrieval": None,
                        "r_video":     None,
                        "r_format":    round(r_format, 4),
                        "n_shots":     0,
                        "text":        normalized_completions[i],
                    })
                continue

            r_video = completion_video_scores[i]

            if q_end > q_start and self.args.dashscope_api_key:
                ep_key = f"{drama_name[i]}/{episode_idx[i]}" if drama_name and episode_idx else ""
                r_retrieval = compute_retrieval_reward(
                    self.reward_model,
                    all_raw_embs[q_start:q_end],
                    ep_key,
                    self.args.top_k_sim,
                    self.args.reward_device,
                )
            else:
                r_retrieval = 0.5
                ep_key = ""

            reward = a * r_retrieval + b * r_video + g * r_format
            rewards.append(float(reward))

            if should_log_completions:
                log_entries.append({
                    "ep_key":      ep_key,
                    "reward":      round(reward, 4),
                    "r_retrieval": round(r_retrieval, 4),
                    "r_video":     round(r_video, 4),
                    "r_format":    round(r_format, 4),
                    "n_shots":     len(shots),
                    "text":        normalized_completions[i],
                })

            if i == 0:
                logger.debug(
                    f"step={self._step} α={a:.2f} [split={split_name}] "
                    f"R={reward:.3f} "
                    f"(ret={r_retrieval:.3f}∈[0,1], vid={r_video:.3f}∈[0,1], fmt={r_format:.3f}∈[-1,1])"
                )

        t_merge = _t() - t0

        # ── 5. 追加写入 completion 观察文件（仅 rank0）──────────────────
        if should_log_completions and log_entries:
            log_path = Path(self.args.output_dir) / "completions_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"step": self._step, "alpha": round(a, 4), "beta": round(b, 4), "completions": log_entries},
                    ensure_ascii=False,
                ) + "\n")

        # ── 耗时汇总：用 print 确保多卡环境下一定可见 ──────────────────
        n_shots = sum(len(s) for s in all_shots_list)
        print(
            f"[reward timing] step={self._step} "
            f"parse={t_parse:.2f}s "
            f"emb_api={t_emb:.2f}s({len(all_queries)}q) "
            f"judge_api={t_judge:.2f}s({len(completion_narratives)}completions) "
            f"retrieval+merge={t_merge:.2f}s "
            f"total={t_parse+t_emb+t_judge+t_merge:.2f}s "
            f"trimmed={repaired_completions} "
            f"shots={n_shots}",
            flush=True,
        )

        return rewards


# ══════════════════════════════════════════════════════════════════════════════
# 数据集加载：seq_continue-only JSON（字段结构与 SFT 原始 JSON 保持一致）
# ══════════════════════════════════════════════════════════════════════════════

def _build_plot_episode_map() -> Dict[str, Tuple[str, str]]:
    """扫描 saved/ 目录，建立 plot_summary → (drama_name, episode_idx) 映射。

    SFT 数据由 saved/ 构建而来，plot_summary 是唯一标识 episode 的字段，
    通过此映射可将 SFT 样本关联到对应 episode 的 depth/pose embedding。
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    for drama_dir in SAVED_DIR.iterdir():
        if not drama_dir.is_dir() or drama_dir.name.startswith("."):
            continue
        plot_dir = drama_dir / "plot"
        if not plot_dir.exists():
            continue
        for pf in plot_dir.glob("*.json"):
            try:
                with open(pf, encoding="utf-8") as f:
                    data = json.load(f)
                ps = data.get("plot_summary", "").strip()
                if ps:
                    mapping[ps] = (drama_dir.name, pf.stem)
            except Exception:
                pass
    logger.info(f"plot_summary → episode 映射：{len(mapping)} 条")
    return mapping


def _collect_needed_ep_keys(
    data_files: List[Path],
    plot_ep_map: Dict[str, Tuple[str, str]],
) -> set:
    """轻量扫描多个数据文件，收集实际用到的 episode key（"drama/ep"）集合。

    用于 preload_all_image_vecs 的 ep_keys_filter，避免把所有 episode 的
    image_vec 全部加载进显存，只加载训练/验证集真正出现的 episode。
    """
    keys: set = set()
    for data_file in data_files:
        if not data_file.exists():
            continue
        raw = load_dataset("json", data_files=str(data_file), split="train")
        for item in raw:
            ps = _extract_plot_summary(item.get("user", ""))
            drama, ep = plot_ep_map.get(ps, ("", ""))
            if drama and ep:
                keys.add(f"{drama}/{ep}")
    logger.info(f"数据集涉及 {len(keys)} 个不重复 episode")
    return keys


def _extract_plot_summary(user_msg: str) -> str:
    """从 SFT user 字段提取 plot_summary（用于与 saved/ 数据匹配）。

    SFT user 字段格式：
        【剧情摘要】\n{plot_summary}\n\n【出场角色】\n...
    """
    m = re.search(r"【剧情摘要】\n(.*?)(?=\n\n【)", user_msg, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback：取 【剧情摘要】 后到第一个双换行
    idx = user_msg.find("【剧情摘要】\n")
    if idx >= 0:
        rest = user_msg[idx + len("【剧情摘要】\n"):]
        end  = rest.find("\n\n")
        return (rest[:end] if end >= 0 else rest).strip()
    return ""


def load_rl_split(
    data_file:   Path,
    tokenizer,
    plot_ep_map: Dict[str, Tuple[str, str]],
    split_name:  str,
    max_samples: Optional[int] = None,
) -> Dataset:
    """加载一个 split，每条样本含 prompt / drama_name / episode_idx / split。"""
    print(f"[dataset] 加载 {data_file.name} ...", flush=True)
    raw = load_dataset("json", data_files=str(data_file), split="train")

    samples: List[dict] = []
    skipped = 0
    for item in raw:
        ps    = _extract_plot_summary(item.get("user", ""))
        drama, ep = plot_ep_map.get(ps, ("", ""))
        ep_key = f"{drama}/{ep}"
        if not drama or ep_key not in _img_vec_cache:
            skipped += 1
            continue
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user",   "content": item["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        samples.append({
            "prompt":      prompt,
            "drama_name":  drama,
            "episode_idx": ep,
            "split":       split_name,
        })
        if max_samples is not None and len(samples) >= max_samples:
            break

    print(
        f"[dataset] {data_file.name}: {len(samples)} 条，"
        f"丢弃 {skipped} 条无 embedding 匹配（{skipped/max(len(samples)+skipped,1)*100:.1f}%）",
        flush=True,
    )
    return Dataset.from_list(samples)


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = HfArgumentParser(RLScriptArgs)
    # return_remaining_strings=True：忽略 torchrun 自动注入的 --local_rank 等参数
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if maybe_relaunch_with_torchrun(args):
        return

    patch_ddp_config_access()

    # API key 优先命令行，其次 config.py
    if not args.dashscope_api_key:
        args.dashscope_api_key = get_dashscope_api_key()
    if not args.judge_api_key:
        args.judge_api_key = get_dashscope_api_key()

    train_data_path = Path(args.train_data_path)
    val_data_path = Path(args.val_data_path)
    test_data_path = Path(args.test_data_path)
    base_model_dir = Path(args.base_model_dir)
    sft_checkpoint_dir = Path(args.sft_checkpoint_dir) if args.sft_checkpoint_dir else None
    resume_from_checkpoint = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    eval_enabled = str(args.eval_strategy).lower() != "no"

    for required_path, path_kind in [
        (base_model_dir, "base_model_dir"),
        (train_data_path, "train_data_path"),
    ]:
        if not required_path.exists():
            raise FileNotFoundError(f"{path_kind} 不存在：{required_path}")
    if eval_enabled and not val_data_path.exists():
        raise FileNotFoundError(f"val_data_path 不存在：{val_data_path}")
    if sft_checkpoint_dir is not None and not sft_checkpoint_dir.exists():
        raise FileNotFoundError(f"sft_checkpoint_dir 不存在：{sft_checkpoint_dir}")
    if resume_from_checkpoint is not None and not resume_from_checkpoint.exists():
        raise FileNotFoundError(f"resume_from_checkpoint 不存在：{resume_from_checkpoint}")

    args.vllm_server_base_url = normalize_vllm_server_base_url(
        args.vllm_server_host,
        args.vllm_server_port,
        args.vllm_server_base_url,
    )
    warmup_steps, warmup_ratio = resolve_warmup_args(args.warmup_steps)

    # ── 构造 GRPOConfig（所有训练超参数显式可见，与 qwen3_5_vision_grpo.py 风格一致）
    # 这里只切换多卡启动方式，训练逻辑保持不变：依旧走 Unsloth + TRL GRPO。
    grpo_config = GRPOConfig(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.num_train_epochs,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size  = args.per_device_eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_generations             = args.num_generations,
        max_prompt_length           = 4600,
        max_completion_length       = args.max_completion_length,
        temperature                 = 0.9,                # 生成 completion 时更偏探索，但仍保持结构稳定
        top_p                       = 0.9,
        repetition_penalty          = 1.2,
        learning_rate               = args.learning_rate,
        warmup_steps                = warmup_steps,
        warmup_ratio                = warmup_ratio,
        weight_decay                = args.weight_decay,
        lr_scheduler_type           = args.lr_scheduler_type,
        logging_steps               = args.logging_steps,
        eval_strategy               = args.eval_strategy,
        eval_steps                  = args.eval_steps,
        save_steps                  = args.save_steps,
        save_total_limit            = args.save_total_limit,
        generation_batch_size       = args.generation_batch_size,
        bf16                        = True,               # 与 SFT 保持一致
        optim                       = "adamw_torch_fused",
        max_grad_norm               = 1.0,                # 梯度裁剪，防止坏 batch 造成梯度爆炸
        use_vllm                    = args.use_vllm,
        vllm_mode                   = args.vllm_mode,
        vllm_server_base_url        = args.vllm_server_base_url,
        vllm_server_host            = args.vllm_server_host,
        vllm_server_port            = args.vllm_server_port,
        vllm_server_timeout         = args.vllm_server_timeout,
        vllm_tensor_parallel_size   = args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization,
    )
    mixed_precision = sync_unsloth_mixed_precision_env(grpo_config)

    if is_main:
        logger.info(f"Running on {num_gpus} GPU(s) with torchrun/DDP")
    logger.info("=== 分镜 RL 训练（GRPO）===")
    logger.info(f"base_model={base_model_dir}")
    logger.info(f"sft_checkpoint={sft_checkpoint_dir or '<none>'}")
    logger.info(f"resume_from_checkpoint={resume_from_checkpoint or '<none>'}")
    logger.info(f"train_data={train_data_path}")
    logger.info(f"val_data={val_data_path if eval_enabled else '<disabled>'}")
    logger.info(f"test_data={test_data_path}")
    logger.info(f"mixed_precision={mixed_precision}")
    logger.info(f"eval_strategy={args.eval_strategy} | save_steps={args.save_steps}")
    logger.info("generation params: temperature=0.9 | top_p=0.9 | repetition_penalty=1.2")
    if args.use_vllm:
        logger.info(
            "vLLM: enabled | mode=%s | server=%s | tp=%s | generation_batch_size=%s",
            args.vllm_mode,
            args.vllm_server_base_url,
            args.vllm_tensor_parallel_size,
            args.generation_batch_size,
        )
    else:
        logger.info("vLLM: disabled")
    logger.info(
        f"奖励：α={args.alpha_start}（检索，固定），"
        f"β {args.beta_start}→{args.beta_end}（LLM-Judge，衰减），γ={args.gamma}（格式）"
    )
    if not args.dashscope_api_key:
        logger.warning("DASHSCOPE_API_KEY 未设置 → R_retrieval = 0，R_video_gen = 0.5（中性）")

    # ── 加载语言模型（base model + LoRA checkpoint 作为 RL 起点）─────────────
    llm_device = resolve_llm_device()
    llm_device_map = {"": llm_device}
    logger.info(f"LLM 设备：{llm_device}")
    logger.info(f"加载 base model {base_model_dir} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_model_dir),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        dtype=torch.bfloat16,
        full_finetuning=False,
        local_files_only=True,
        device_map=llm_device_map,
    )

    if sft_checkpoint_dir is not None:
        patch_peft_tensor_parallel_compat()
        from peft import PeftModel

        logger.info(f"加载 LoRA checkpoint {sft_checkpoint_dir} 作为 RL 初始化权重 ...")
        model = PeftModel.from_pretrained(
            model,
            str(sft_checkpoint_dir),
            is_trainable=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        model = FastLanguageModel.patch_peft_model(
            model,
            use_gradient_checkpointing="unsloth",
        )
    if llm_device != "cpu":
        model = model.to(llm_device)

    # GRPO 需要 left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 加载已训练好的 Reward Model（冻结，仅做推理）────────────────────────
    # ★ 使用 best_model.pt 中已学习好的投影权重（text_proj / depth_proj / pose_proj /
    #   image_fuser），RL 训练期间其参数不更新，只有 LLM 的 LoRA 权重被优化。
    #   文本侧 encode_text() 和图像侧 encode_images() 均使用这套已训练权重，
    #   确保两侧在同一个 2048-dim 对齐空间内做余弦相似度比较。

    # reward_device：每个 rank 绑定自身的 GPU，避免 CPU↔GPU 搬运开销。
    # "auto" → cuda:{local_rank}；也可命令行显式指定 --reward_device cpu 作为回退。
    if args.reward_device == "auto":
        args.reward_device = f"cuda:{local_rank}"
    logger.info(f"Reward model 设备：{args.reward_device}")

    logger.info("加载已训练的 StoryboardRewardModel（冻结参数）...")
    reward_model = StoryboardRewardModel(emb_dim=args.emb_dim, hidden_dim=args.hidden_dim)
    ckpt = torch.load(REWARD_CKPT, map_location=args.reward_device)
    reward_model.load_state_dict(ckpt["model_state"])
    reward_model.eval()
    for p in reward_model.parameters():     # 显式冻结：确保 reward model 参数不参与梯度
        p.requires_grad_(False)
    reward_model.to(args.reward_device)
    logger.info(
        f"  epoch={ckpt.get('epoch','?')}, "
        f"val Recall@10={ckpt.get('val_metrics',{}).get('recall_10',0):.3f} "
        f"（参数已冻结，仅用于推理）"
    )

    # ── 建立 plot_summary → episode 映射，收集各 split 的 episode keys ──
    plot_ep_map = _build_plot_episode_map()
    train_ep_keys = _collect_needed_ep_keys([train_data_path], plot_ep_map)
    split_ep_keys = {"train": train_ep_keys}
    if eval_enabled:
        split_ep_keys["val"] = _collect_needed_ep_keys([val_data_path], plot_ep_map)

    # ── 构建各 split 的全量帧矩阵（首次计算后缓存到文件，后续直接加载）──
    preload_split_image_vecs(
        split_ep_keys=split_ep_keys,
        reward_model=reward_model,
        device=args.reward_device,
    )

    # ── 加载数据集 ────────────────────────────────────────────────────────
    train_dataset = load_rl_split(train_data_path, tokenizer, plot_ep_map, "train", args.max_train_samples)
    eval_dataset = None
    if eval_enabled:
        eval_dataset = load_rl_split(val_data_path, tokenizer, plot_ep_map, "val", args.max_eval_samples)

    # ── 估算总步数（α 课程学习调度用） ───────────────────────────────────
    bsz = grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps
    # 多卡时每个 step 所有卡合并处理 bsz × num_gpus 条样本
    steps_per_epoch = max(len(train_dataset) // max(bsz * num_gpus, 1), 1)
    total_steps = steps_per_epoch * int(grpo_config.num_train_epochs)

    # ── 构建 Reward Orchestrator ──────────────────────────────────────────
    reward_fn = RewardOrchestrator(args, reward_model, total_steps=total_steps)

    # ── GRPO 训练 ─────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    logger.info("开始 GRPO 训练 ...")
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)

    # ── 保存 RL 微调后的模型 ──────────────────────────────────────────────
    # 等所有进程完成训练后再保存（避免 rank>0 的进程提前退出）
    if dist.is_initialized():
        dist.barrier()
    save_path = Path(grpo_config.output_dir) / "final_rl_lora"
    # 统一走 trainer.save_model()，保持和原脚本一致的保存路径与行为。
    trainer.save_model(str(save_path))
    if is_main:
        tokenizer.save_pretrained(str(save_path))
        logger.info(f"RL 微调模型已保存至 {save_path}")


if __name__ == "__main__":
    main()
