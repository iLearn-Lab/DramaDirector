import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_dashscope_api_key
from project_paths import DEFAULT_DEPTH_EMB_DIR, DEFAULT_POSE_EMB_DIR

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
U_ESCAPE_PATTERN = re.compile(r"#U([0-9a-fA-F]{4,6})")
SCRIPT_VERSION = "2026-04-23.package_outputs_v2"
MAX_RETRIES = 3


@dataclass
class IndexItem:
    depth_path: Path
    pose_path: Path
    drama_name: str
    episode_idx: str
    keyframe_name: str


@dataclass
class QueryShot:
    source_file: Path
    sample_index: int
    shot_index: Optional[int]
    prompt: str
    duration: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto retrieval check: prefer reward-model retrieval when a checkpoint is found, "
            "otherwise fall back to baseline (mean(depth, pose)) retrieval."
        )
    )
    parser.add_argument(
        "--video_inputs_glob",
        type=str,
        default="prepared_video_inputs/video_inputs/infer_results_*/sample_*.json",
        help="Glob for prepared video input JSON files (used when --input_dir is empty).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Input directory containing sample json files. If set, takes precedence over --video_inputs_glob.",
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        default="infer_results_*/sample_*.json",
        help="Glob pattern relative to --input_dir when --input_dir is set.",
    )
    parser.add_argument("--depth_emb_dir", type=str, default=str(DEFAULT_DEPTH_EMB_DIR))
    parser.add_argument("--pose_emb_dir", type=str, default=str(DEFAULT_POSE_EMB_DIR))
    parser.add_argument("--depth_image_dir", type=str, default="depth")
    parser.add_argument("--pose_image_dir", type=str, default="pose")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="How many sample files to test. <=0 means use all matched sample files.",
    )
    parser.add_argument(
        "--shots_per_sample",
        type=int,
        default=2,
        help="How many shots per sample file. <=0 means use all shots in each selected sample.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--index_limit",
        type=int,
        default=0,
        help="Optional cap on indexed depth/pose items for faster dry-runs. 0 means all.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding API calls.")
    parser.add_argument("--embed_model", type=str, default="tongyi-embedding-vision-flash-2026-03-06")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="auto",
        choices=["auto", "reward_only", "baseline_only"],
        help=(
            "auto: prefer reward model and fallback to baseline; "
            "reward_only: use reward model only; "
            "baseline_only: use baseline only."
        ),
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Optional explicit reward-model checkpoint path (.pt/.pth/.ckpt).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="reward_model",
        help="Directory for auto-discovering reward-model checkpoints.",
    )
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=1024,
        help="Batch size for reward-model index/query encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for reward-model encoding.",
    )
    parser.add_argument(
        "--strict_reward_model",
        action="store_true",
        help="If set, do not fall back to baseline when reward-model loading/encoding fails.",
    )

    parser.add_argument("--output_file", type=str, default="retrieval_check_results.auto.json")
    parser.add_argument("--vlm_seed_file", type=str, default="retrieval_check_vlm_seed.auto.jsonl")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="If set, place output files under this directory (unless output paths are absolute).",
    )
    parser.add_argument(
        "--path_mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="Path serialization mode in JSON outputs.",
    )
    parser.add_argument(
        "--path_base",
        type=str,
        default=".",
        help="Base directory used when --path_mode=relative.",
    )
    return parser.parse_args()


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        normalized[nk] = v
    return normalized


def find_weight_shape_from_state(state_dict: dict[str, Any], key_suffix: str) -> Optional[tuple[int, ...]]:
    if key_suffix in state_dict:
        return tuple(state_dict[key_suffix].shape)
    for k, v in state_dict.items():
        if k.endswith(key_suffix):
            return tuple(v.shape)
    return None


def discover_checkpoint(explicit_path: str, checkpoint_dir: str) -> tuple[Optional[Path], str]:
    if explicit_path.strip():
        p = Path(explicit_path.strip())
        if p.exists():
            return p, "explicit"
        return None, f"explicit checkpoint not found: {p}"

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None, f"checkpoint dir not found: {ckpt_dir}"

    preferred = [
        ckpt_dir / "best_model.pt",
        ckpt_dir / "final_model.pt",
    ]
    for p in preferred:
        if p.exists():
            return p, "preferred"

    wildcard_candidates: list[Path] = []
    for pattern in ["final_model*.pt", "*.pt", "*.pth", "*.ckpt"]:
        wildcard_candidates.extend(sorted(ckpt_dir.glob(pattern)))

    if not wildcard_candidates:
        return None, f"no checkpoint files found under: {ckpt_dir}"

    wildcard_candidates = sorted(wildcard_candidates, key=lambda x: x.stat().st_mtime, reverse=True)
    return wildcard_candidates[0], "auto_latest"


def choose_device(torch_module, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def iter_batches(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(texts: list[str], api_key: str, model: str, batch_size: int) -> np.ndarray:
    try:
        import dashscope
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "dashscope is required for text embedding. Install it with: pip install dashscope"
        ) from e

    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is empty.")
    if not api_key.isascii():
        raise ValueError(
            "DASHSCOPE_API_KEY contains non-ASCII characters. "
            "It looks like a placeholder (e.g. Chinese text) or malformed key."
        )

    all_embeddings: list[list[float]] = []

    for chunk in iter_batches(texts, batch_size):
        payload = [{"text": t} for t in chunk]
        last_error: Optional[str] = None
        for attempt in range(1, MAX_RETRIES + 1):
            resp = dashscope.MultiModalEmbedding.call(
                api_key=api_key,
                model=model,
                input=payload,
            )
            if resp.status_code == 200:
                ordered = sorted(resp.output["embeddings"], key=lambda x: x["index"])
                all_embeddings.extend([e["embedding"] for e in ordered])
                break
            last_error = f"status={resp.status_code} message={resp.message}"
            if attempt < MAX_RETRIES:
                time.sleep(attempt)
        else:
            raise RuntimeError(f"Embedding API failed after retries: {last_error}")

    arr = np.array(all_embeddings, dtype=np.float32)
    return normalize_rows(arr)


def topk(sim: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, sim.shape[1])
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    row_ids = np.arange(sim.shape[0])[:, None]
    vals = sim[row_ids, idx]
    order = np.argsort(-vals, axis=1)
    sorted_idx = idx[row_ids, order]
    sorted_vals = vals[row_ids, order]
    return sorted_idx, sorted_vals


def collect_index_items(depth_dir: Path, pose_dir: Path, index_limit: int) -> list[IndexItem]:
    depth_files = sorted(depth_dir.rglob("keyframe_*.npy"))
    if not depth_files:
        raise FileNotFoundError(f"No depth embedding files found under: {depth_dir}")

    items: list[IndexItem] = []
    for depth_path in depth_files:
        rel = depth_path.relative_to(depth_dir)
        pose_path = pose_dir / rel
        if not pose_path.exists():
            continue

        parts = rel.parts
        drama_name = parts[0] if len(parts) > 0 else ""
        episode_idx = parts[1] if len(parts) > 1 else ""
        keyframe_name = parts[-1]
        items.append(
            IndexItem(
                depth_path=depth_path,
                pose_path=pose_path,
                drama_name=drama_name,
                episode_idx=episode_idx,
                keyframe_name=keyframe_name,
            )
        )

    if not items:
        raise RuntimeError("No paired depth/pose embeddings found.")

    if index_limit > 0 and len(items) > index_limit:
        random.shuffle(items)
        items = items[:index_limit]
    return items


def collect_query_shots_flexible(
    video_input_paths: list[Path],
    num_samples: int,
    shots_per_sample: int,
) -> list[QueryShot]:
    chosen_files = list(video_input_paths)
    random.shuffle(chosen_files)

    if num_samples > 0:
        chosen_files = chosen_files[: min(num_samples, len(chosen_files))]

    queries: list[QueryShot] = []
    for path in chosen_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        sample_index = int(data.get("sample_index", 0))
        shots = data.get("shots", [])
        if not shots:
            continue

        shot_indices = list(range(len(shots)))
        random.shuffle(shot_indices)
        if shots_per_sample > 0:
            shot_indices = shot_indices[: min(shots_per_sample, len(shots))]

        for si in shot_indices:
            shot = shots[si]
            prompt = (shot.get("prompt") or "").strip()
            if not prompt:
                continue
            duration = shot.get("duration")
            try:
                duration = float(duration) if duration is not None else None
            except (TypeError, ValueError):
                duration = None
            queries.append(
                QueryShot(
                    source_file=path,
                    sample_index=sample_index,
                    shot_index=shot.get("shot_index"),
                    prompt=prompt,
                    duration=duration,
                )
            )
    return queries


def build_baseline_image_vecs_from_items(items: list[IndexItem]) -> np.ndarray:
    if not items:
        raise RuntimeError("No index items provided.")

    first_depth = np.load(str(items[0].depth_path)).astype(np.float32)
    first_pose = np.load(str(items[0].pose_path)).astype(np.float32)
    if first_depth.shape != first_pose.shape:
        raise ValueError(
            "Depth/Pose embedding shape mismatch for first item: "
            f"{items[0].depth_path}={first_depth.shape}, {items[0].pose_path}={first_pose.shape}"
        )
    emb_dim = int(first_depth.shape[0])

    image_vecs = np.empty((len(items), emb_dim), dtype=np.float32)
    for i, it in enumerate(items):
        depth_vec = np.load(str(it.depth_path)).astype(np.float32)
        pose_vec = np.load(str(it.pose_path)).astype(np.float32)
        if depth_vec.shape != pose_vec.shape:
            raise ValueError(
                "Depth/Pose embedding shape mismatch: "
                f"{it.depth_path}={depth_vec.shape}, {it.pose_path}={pose_vec.shape}"
            )
        if depth_vec.shape[0] != emb_dim:
            raise ValueError(
                f"Inconsistent embedding dimension in index: expected {emb_dim}, "
                f"but got {depth_vec.shape[0]} at {it.depth_path}"
            )
        image_vecs[i] = (depth_vec + pose_vec) / 2.0
        if (i + 1) % 10000 == 0:
            print(f"[Baseline] Loaded {i + 1}/{len(items)}")
    return normalize_rows(image_vecs)


def decode_u_escapes(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        code = int(match.group(1), 16)
        try:
            return chr(code)
        except ValueError:
            return match.group(0)

    return U_ESCAPE_PATTERN.sub(_repl, text)


def resolve_image_path(emb_path: Path, emb_root: Path, image_root: Path) -> Optional[Path]:
    if not image_root.exists():
        return None
    try:
        rel = emb_path.relative_to(emb_root)
    except ValueError:
        return None

    rel_variants = [rel]
    if len(rel.parts) >= 2 and rel.parts[0] == emb_root.name:
        rel_variants.append(Path(*rel.parts[1:]))
    if len(rel.parts) >= 2 and rel.parts[0] in {"depth_emb", "pose_emb"}:
        rel_variants.append(Path(*rel.parts[1:]))

    for rel_item in rel_variants:
        stem_rel = rel_item.with_suffix("")
        part_variants = {
            tuple(stem_rel.parts),
            tuple(decode_u_escapes(p) for p in stem_rel.parts),
        }
        for parts in part_variants:
            stem_str = str(Path(*parts))
            candidates = [stem_str, f"{stem_str}_generated"]
            for base in candidates:
                for ext in IMAGE_EXTS:
                    candidate = image_root / f"{base}{ext}"
                    if candidate.exists():
                        return candidate
                for ext in IMAGE_EXTS:
                    candidate = image_root / f"{base}{ext.upper()}"
                    if candidate.exists():
                        return candidate
    return None


def build_vlm_prompt(shot_prompt: str, depth_image: Optional[Path], pose_image: Optional[Path]) -> str:
    control_hint = []
    if depth_image:
        control_hint.append("使用图1深度图控制图约束空间结构")
    if pose_image:
        control_hint.append("使用图2骨架图控制图约束人物姿态，必须遵守着两条约束条件")
    hint = "，".join(control_hint) if control_hint else "无控制图，仅基于文本"
    return (
        "请生成该镜头的首帧高质量图像。\n"
        f"镜头描述：{shot_prompt}\n"
        f"控制条件：{hint}\n"
        "要求：写实风格、主体清晰、构图稳定，便于后续视频模型以该首帧继续生成。"
    )


def build_task_id(source_file: Path, sample_index: int, shot_index: Optional[int], serial_idx: int) -> str:
    src = source_file.parent.name + "_" + source_file.stem
    if shot_index is None:
        shot = serial_idx
    else:
        try:
            shot = int(shot_index)
        except (TypeError, ValueError):
            shot = serial_idx
    return f"{src}_s{sample_index:03d}_k{int(shot):03d}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def serialize_path(path_obj: Optional[Path], mode: str, base_dir: Path) -> Optional[str]:
    if path_obj is None:
        return None
    p = Path(path_obj).expanduser()
    if mode == "absolute":
        if p.is_absolute():
            return str(p)
        return str((Path.cwd() / p).resolve())

    # relative mode
    if not p.is_absolute():
        return str(p)
    return os.path.relpath(str(p), str(base_dir))


def get_script_fingerprint() -> dict[str, str]:
    script_path = Path(__file__).resolve()
    content = script_path.read_bytes()
    return {
        "script_path": str(script_path),
        "script_sha256": hashlib.sha256(content).hexdigest(),
        "script_version": SCRIPT_VERSION,
    }


def load_reward_model(checkpoint_path: Path, requested_device: str):
    try:
        import torch
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyTorch is required for reward-model retrieval. Install torch or allow baseline fallback."
        ) from e

    from reward_model.model import StoryboardRewardModel

    device = choose_device(torch, requested_device)
    raw_obj = torch.load(str(checkpoint_path), map_location="cpu")

    ckpt_args: dict[str, Any] = {}
    state_dict: Optional[dict[str, Any]] = None

    if isinstance(raw_obj, dict):
        for key in ["model_state", "state_dict", "model"]:
            if key in raw_obj and isinstance(raw_obj[key], dict):
                state_dict = raw_obj[key]
                break
        if state_dict is None:
            tensor_like = [k for k, v in raw_obj.items() if hasattr(v, "shape")]
            if tensor_like:
                state_dict = raw_obj
        if "args" in raw_obj and isinstance(raw_obj["args"], dict):
            ckpt_args = raw_obj["args"]

    if state_dict is None:
        raise RuntimeError(
            "Unsupported checkpoint format. Expected a state_dict or dict containing model_state/state_dict."
        )

    state_dict = normalize_state_dict_keys(state_dict)

    shape = find_weight_shape_from_state(state_dict, "text_proj.proj.0.weight")
    if shape is None or len(shape) != 2:
        raise RuntimeError("Cannot infer (hidden_dim, emb_dim) from checkpoint key: text_proj.proj.0.weight")

    hidden_dim, emb_dim = int(shape[0]), int(shape[1])
    model = StoryboardRewardModel(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_img_heads=int(ckpt_args.get("num_img_heads", 4)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_prefixes = ("text_proj.", "depth_proj.", "pose_proj.", "image_fuser.")
    missing_critical = [k for k in missing if k.startswith(critical_prefixes)]
    if missing_critical:
        raise RuntimeError(
            "Checkpoint misses critical retrieval weights: "
            + ", ".join(missing_critical[:8])
            + (" ..." if len(missing_critical) > 8 else "")
        )

    model.eval().to(device)

    meta = {
        "device": device,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return model, meta


def encode_index_with_reward_model(items: list[Any], model, device: str, batch_size: int) -> np.ndarray:
    import torch

    all_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            depth = np.stack([np.load(str(it.depth_path)).astype(np.float32) for it in chunk], axis=0)
            pose = np.stack([np.load(str(it.pose_path)).astype(np.float32) for it in chunk], axis=0)

            depth_t = torch.from_numpy(depth).to(device=device, dtype=torch.float32)
            pose_t = torch.from_numpy(pose).to(device=device, dtype=torch.float32)
            _, image_vec = model.encode_images(depth_t, pose_t)
            all_vecs.append(image_vec.detach().cpu().numpy().astype(np.float32))

            done = min(start + batch_size, len(items))
            if done % 10000 == 0 or done == len(items):
                print(f"[RewardModel] Encoded index items: {done}/{len(items)}")

    return np.concatenate(all_vecs, axis=0)


def encode_queries_with_reward_model(text_embs: np.ndarray, model, device: str, batch_size: int) -> np.ndarray:
    import torch

    all_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, text_embs.shape[0], batch_size):
            chunk = text_embs[start : start + batch_size]
            text_t = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
            text_vec = model.encode_text(text_t)
            all_vecs.append(text_vec.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(all_vecs, axis=0)


def main() -> None:
    args = parse_args()
    fingerprint = get_script_fingerprint()
    path_base_dir = Path(args.path_base).expanduser().resolve()
    print(
        "[Script] "
        f"version={fingerprint['script_version']} "
        f"path={fingerprint['script_path']} "
        f"sha256={fingerprint['script_sha256']}"
    )

    api_key = args.api_key or get_dashscope_api_key()
    if not api_key:
        raise ValueError("Missing API key. Please set DASHSCOPE_API_KEY in config.py or pass --api_key.")

    if args.input_dir.strip():
        input_dir = Path(args.input_dir).expanduser()
        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        input_pattern = args.input_glob
        video_input_paths = sorted(input_dir.glob(input_pattern))
        input_source = str(input_dir / input_pattern)
    else:
        video_input_paths = sorted(Path(".").glob(args.video_inputs_glob))
        input_source = args.video_inputs_glob

    if not video_input_paths:
        raise FileNotFoundError(f"No files matched: {input_source}")

    print(f"[Query] Found {len(video_input_paths)} prepared sample files.")
    queries = collect_query_shots_flexible(
        video_input_paths=video_input_paths,
        num_samples=args.num_samples,
        shots_per_sample=args.shots_per_sample,
    )
    if not queries:
        raise RuntimeError("No query shots collected from video inputs.")
    print(f"[Query] Collected {len(queries)} shot prompts for retrieval checks.")

    depth_dir = Path(args.depth_emb_dir)
    pose_dir = Path(args.pose_emb_dir)
    depth_image_dir = Path(args.depth_image_dir)
    pose_image_dir = Path(args.pose_image_dir)
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth_emb directory not found: {depth_dir}")
    if not pose_dir.exists():
        raise FileNotFoundError(f"pose_emb directory not found: {pose_dir}")
    if not depth_image_dir.exists():
        print(f"[WARN] depth image directory not found: {depth_image_dir}")
    if not pose_image_dir.exists():
        print(f"[WARN] pose image directory not found: {pose_image_dir}")

    print("[Index] Collecting paired depth/pose candidates ...")
    items = collect_index_items(
        depth_dir=depth_dir,
        pose_dir=pose_dir,
        index_limit=args.index_limit,
    )
    print(f"[Index] Ready: {len(items)} paired candidates.")

    query_texts = [q.prompt for q in queries]
    print("[Embed] Calling text embedding API ...")
    text_embs = embed_texts(
        texts=query_texts,
        api_key=api_key,
        model=args.embed_model,
        batch_size=args.batch_size,
    )
    print(f"[Embed] Got {text_embs.shape[0]} query embeddings.")

    backend = "baseline"
    reward_checkpoint = None
    reward_device = None
    fallback_reason = ""
    discover_mode = ""

    query_vecs = text_embs
    image_vecs: Optional[np.ndarray] = None

    def set_baseline_vectors(reason: str = "") -> None:
        nonlocal image_vecs, query_vecs, backend, fallback_reason
        print("[Baseline] Building baseline retrieval vectors ...")
        image_vecs = build_baseline_image_vecs_from_items(items)
        if text_embs.shape[1] != image_vecs.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch between query texts and depth/pose index: "
                f"text_dim={text_embs.shape[1]}, index_dim={image_vecs.shape[1]}. "
                "Use the same Tongyi model for both sides, e.g. "
                "`--embed_model tongyi-embedding-vision-flash` when your depth/pose embeddings are flash."
            )
        backend = "baseline"
        query_vecs = text_embs
        fallback_reason = reason

    if args.retrieval_mode == "baseline_only":
        set_baseline_vectors("")
    else:
        ckpt_path, discover_mode = discover_checkpoint(args.checkpoint_path, args.checkpoint_dir)
        if ckpt_path is None:
            msg = f"[RewardModel] {discover_mode}."
            if args.retrieval_mode == "reward_only":
                raise RuntimeError(msg + " reward_only mode does not allow baseline fallback.")
            print(msg + " Use baseline retrieval.")
            set_baseline_vectors(discover_mode)
            reward_checkpoint = None
            reward_device = None
        else:
            reward_checkpoint = str(ckpt_path)
            print(f"[RewardModel] Candidate checkpoint found ({discover_mode}): {ckpt_path}")
            try:
                model, model_meta = load_reward_model(ckpt_path, args.device)
                reward_device = model_meta["device"]
                model_emb_dim = int(model_meta["emb_dim"])
                if model_emb_dim != text_embs.shape[1]:
                    raise ValueError(
                        f"Reward model emb_dim={model_emb_dim}, but query/index embedding dim={text_embs.shape[1]}. "
                        "Please use matching Tongyi embedding model for both index and text query."
                    )

                print(f"[RewardModel] Loaded. device={reward_device} emb_dim={model_emb_dim} hidden_dim={model_meta['hidden_dim']}")
                image_vecs = encode_index_with_reward_model(items, model, reward_device, args.reward_batch_size)
                query_vecs = encode_queries_with_reward_model(text_embs, model, reward_device, args.reward_batch_size)
                backend = "reward_model"
                print("[RewardModel] Retrieval vectors ready.")
            except Exception as e:
                if args.retrieval_mode == "reward_only" or args.strict_reward_model:
                    raise
                reason = str(e)
                print(f"[RewardModel][WARN] Failed, fallback to baseline: {reason}")
                set_baseline_vectors(reason)

    if image_vecs is None:
        raise RuntimeError("Internal error: retrieval vectors are not initialized.")

    sim = query_vecs @ image_vecs.T
    top_idx, top_vals = topk(sim, args.top_k)

    results: list[dict[str, Any]] = []
    vlm_seed_rows: list[dict[str, Any]] = []
    retrieval_rows_package: list[dict[str, Any]] = []
    vlm_tasks: list[dict[str, Any]] = []
    video_shot_tasks: list[dict[str, Any]] = []
    package_base_dir = Path(args.output_dir).expanduser() if args.output_dir.strip() else Path(".")
    for qi, query in enumerate(queries):
        retrieved = []
        query_topk_package: list[dict[str, Any]] = []
        query_source_path = serialize_path(query.source_file, args.path_mode, path_base_dir)
        for rank, (ii, score) in enumerate(zip(top_idx[qi], top_vals[qi]), start=1):
            item = items[int(ii)]
            depth_img = resolve_image_path(item.depth_path, depth_dir, depth_image_dir)
            pose_img = resolve_image_path(item.pose_path, pose_dir, pose_image_dir)
            cand = {
                "rank": rank,
                "score": float(score),
                "drama_name": item.drama_name,
                "episode_idx": item.episode_idx,
                "keyframe_name": item.keyframe_name,
                "depth_emb_path": serialize_path(item.depth_path, args.path_mode, path_base_dir),
                "pose_emb_path": serialize_path(item.pose_path, args.path_mode, path_base_dir),
                "depth_image_path": serialize_path(depth_img, args.path_mode, path_base_dir),
                "pose_image_path": serialize_path(pose_img, args.path_mode, path_base_dir),
            }
            retrieved.append(cand)
            query_topk_package.append(cand)

        top1 = retrieved[0]
        top1_depth_image = top1.get("depth_image_path")
        top1_pose_image = top1.get("pose_image_path")

        query_pkg = {
            "source_file": query_source_path,
            "sample_index": query.sample_index,
            "shot_index": query.shot_index,
            "duration": query.duration,
            "prompt": query.prompt,
            "topk": query_topk_package,
        }
        retrieval_rows_package.append(query_pkg)

        depth_img_path = Path(top1_depth_image) if top1_depth_image else None
        pose_img_path = Path(top1_pose_image) if top1_pose_image else None
        task_id = build_task_id(
            source_file=query.source_file,
            sample_index=query.sample_index,
            shot_index=query.shot_index,
            serial_idx=qi + 1,
        )
        first_frame_path = package_base_dir / "first_frames" / f"{task_id}.png"
        clip_path = package_base_dir / "video_clips" / f"{task_id}.mp4"
        first_frame_out = serialize_path(first_frame_path, args.path_mode, path_base_dir)
        clip_out = serialize_path(clip_path, args.path_mode, path_base_dir)
        status = "ready" if (depth_img_path is not None and pose_img_path is not None) else "missing_controls"
        vlm_prompt = build_vlm_prompt(query.prompt, depth_img_path, pose_img_path)

        vlm_tasks.append(
            {
                "task_id": task_id,
                "status": status,
                "source_file": query_source_path,
                "sample_index": query.sample_index,
                "shot_index": query.shot_index,
                "duration": query.duration,
                "retrieval_score": top1["score"],
                "drama_name": top1["drama_name"],
                "episode_idx": top1["episode_idx"],
                "keyframe_name": top1["keyframe_name"],
                "shot_prompt": query.prompt,
                "depth_control_image": top1_depth_image,
                "pose_control_image": top1_pose_image,
                "vlm_prompt": vlm_prompt,
                "expected_first_frame_path": first_frame_out,
            }
        )
        video_shot_tasks.append(
            {
                "task_id": task_id,
                "status": status,
                "duration": query.duration,
                "shot_prompt": query.prompt,
                "first_frame_path": first_frame_out,
                "depth_control_image": top1_depth_image,
                "pose_control_image": top1_pose_image,
                "expected_video_clip_path": clip_out,
                "source_file": query_source_path,
                "sample_index": query.sample_index,
                "shot_index": query.shot_index,
            }
        )
        vlm_seed_rows.append(
            {
                "task_id": task_id,
                "status": status,
                "source_file": query_source_path,
                "sample_index": query.sample_index,
                "shot_index": query.shot_index,
                "shot_prompt": query.prompt,
                "duration": query.duration,
                "retrieved_depth_emb_path": top1["depth_emb_path"],
                "retrieved_pose_emb_path": top1["pose_emb_path"],
                "retrieved_depth_image_path": top1_depth_image,
                "retrieved_pose_image_path": top1_pose_image,
                "depth_control_image": top1_depth_image,
                "pose_control_image": top1_pose_image,
                "vlm_prompt": vlm_prompt,
                "expected_first_frame_path": first_frame_out,
                "retrieval_score": top1["score"],
                "drama_name": top1["drama_name"],
                "episode_idx": top1["episode_idx"],
                "keyframe_name": top1["keyframe_name"],
            }
        )

        results.append(
            {
                "query": {
                    "source_file": query_source_path,
                    "sample_index": query.sample_index,
                    "shot_index": query.shot_index,
                    "duration": query.duration,
                    "prompt": query.prompt,
                },
                "retrieval_topk": retrieved,
            }
        )

    output = {
        "config": {
            "script_version": fingerprint["script_version"],
            "script_path": fingerprint["script_path"],
            "script_sha256": fingerprint["script_sha256"],
            "path_mode": args.path_mode,
            "path_base": str(path_base_dir),
            "video_inputs_glob": args.video_inputs_glob,
            "input_dir": args.input_dir,
            "input_glob": args.input_glob,
            "resolved_input_source": input_source,
            "depth_emb_dir": str(depth_dir),
            "pose_emb_dir": str(pose_dir),
            "depth_image_dir": str(depth_image_dir),
            "pose_image_dir": str(pose_image_dir),
            "num_samples": args.num_samples,
            "shots_per_sample": args.shots_per_sample,
            "top_k": args.top_k,
            "index_limit": args.index_limit,
            "embed_model": args.embed_model,
            "retrieval_mode": args.retrieval_mode,
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_dir": args.checkpoint_dir,
            "reward_batch_size": args.reward_batch_size,
            "device": args.device,
            "strict_reward_model": args.strict_reward_model,
            "output_dir": args.output_dir,
        },
        "retrieval_backend": backend,
        "reward_model_checkpoint": reward_checkpoint,
        "reward_model_device": reward_device,
        "reward_model_fallback_reason": fallback_reason,
        "num_queries": len(queries),
        "num_candidates": len(items),
        "num_vlm_tasks": len(vlm_tasks),
        "num_ready_vlm_tasks": sum(1 for t in vlm_tasks if t.get("status") == "ready"),
        "results": results,
    }

    output_path = Path(args.output_file).expanduser()
    vlm_seed_path = Path(args.vlm_seed_file).expanduser()
    retrieval_topk_path = Path("retrieval_topk.json")
    vlm_tasks_path = Path("vlm_firstframe_tasks.jsonl")
    video_shot_tasks_path = Path("video_shot_tasks.jsonl")
    summary_path = Path("summary.json")
    if args.output_dir.strip():
        out_dir = Path(args.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not output_path.is_absolute():
            output_path = out_dir / output_path
        if not vlm_seed_path.is_absolute():
            vlm_seed_path = out_dir / vlm_seed_path
        retrieval_topk_path = out_dir / retrieval_topk_path
        vlm_tasks_path = out_dir / vlm_tasks_path
        video_shot_tasks_path = out_dir / video_shot_tasks_path
        summary_path = out_dir / summary_path
    else:
        base_dir = output_path.parent
        retrieval_topk_path = base_dir / retrieval_topk_path
        vlm_tasks_path = base_dir / vlm_tasks_path
        video_shot_tasks_path = base_dir / video_shot_tasks_path
        summary_path = base_dir / summary_path

    write_json(output_path, output)
    write_json(retrieval_topk_path, {"summary": output, "results": retrieval_rows_package})
    write_json(summary_path, output)

    write_jsonl(vlm_seed_path, vlm_seed_rows)
    write_jsonl(vlm_tasks_path, vlm_tasks)
    write_jsonl(video_shot_tasks_path, video_shot_tasks)

    print(f"[Done] Backend={backend}")
    if reward_checkpoint:
        print(f"[Done] Checkpoint={reward_checkpoint}")
    if fallback_reason and backend == "baseline":
        print(f"[Done] FallbackReason={fallback_reason}")
    print(f"[Done] Retrieval report: {output_path}")
    print(f"[Done] Package retrieval_topk: {retrieval_topk_path}")
    print(f"[Done] Package vlm tasks: {vlm_tasks_path}")
    print(f"[Done] Package video-shot tasks: {video_shot_tasks_path}")
    print(f"[Done] Package summary: {summary_path}")
    print(f"[Done] VLM rows: {vlm_seed_path}")
    print(f"[Done] Queries={len(queries)}  Candidates={len(items)}  TopK={args.top_k}")


if __name__ == "__main__":
    main()
