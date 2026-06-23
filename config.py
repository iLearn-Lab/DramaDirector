from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESS_ROOT = PROJECT_ROOT / "preprocess"


# -----------------------------------------------------------------------------
# API keys
# Edit these values directly for local experiments.
# -----------------------------------------------------------------------------

# Used by preprocessing, embedding, retrieval checks, and reward-model text encoding.
DASHSCOPE_API_KEY = ""

# Optional multi-key pool for scripts that support concurrent request sharding.
# If empty, scripts fall back to [DASHSCOPE_API_KEY].
DASHSCOPE_API_KEYS: list[str] = []

# Optional RunningHub key for downstream video generation utilities.
RUNNINGHUB_API_KEY = ""


# -----------------------------------------------------------------------------
# Data paths
# Edit these if your local layout differs from the repository defaults.
# -----------------------------------------------------------------------------

SAVED_DIR = PREPROCESS_ROOT / "saved"
PROCESSED_SPLIT_DIR = PREPROCESS_ROOT / "processed_split"
TEXT_EMB_DIR = PREPROCESS_ROOT / "text_emb"
DEPTH_EMB_DIR = PREPROCESS_ROOT / "depth_emb"
POSE_EMB_DIR = PREPROCESS_ROOT / "pose_emb"

# Reward-model scripts expect a base directory that contains text_emb/depth_emb/pose_emb.
EMB_BASE_DIR = PREPROCESS_ROOT


def get_dashscope_api_key() -> str:
    return DASHSCOPE_API_KEY.strip()


def get_dashscope_api_keys() -> list[str]:
    keys = [key.strip() for key in DASHSCOPE_API_KEYS if key and key.strip()]
    if keys:
        return keys
    single = get_dashscope_api_key()
    return [single] if single else []


def get_runninghub_api_key() -> str:
    return RUNNINGHUB_API_KEY.strip()
