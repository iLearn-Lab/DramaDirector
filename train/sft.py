from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401  # Import before transformers/peft so Unsloth patches them first.
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback

from render_storyboard_verl_text_dataset import render_table
from storyboard_short_task_packing import (
    PackGroup,
    build_short_task_packing_summary,
    first_fit_decreasing_pack,
    infer_split_from_paths,
    sort_groups_by_first_index,
)

# python sft_unsloth_multigpu.py --nproc-per-node 4 四卡
SHORT_TASK_TYPES = ("single_shot", "multifield", "ordering", "summary_compress")
RENDER_WAIT_SECONDS = 900
PACKING_CACHE_WAIT_SECONDS = 900
OFFICIAL_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):  # noqa: ARG002
        torch.cuda.empty_cache()


def quote_for_display(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def check_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def resolve_default_lr(finetune_mode: str, explicit_lr: float | None) -> float:
    if explicit_lr is not None:
        return explicit_lr
    return 2e-4 if finetune_mode == "lora" else 1e-5


def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def resolve_model_ref(model_path_arg: str) -> str:
    model_path = Path(model_path_arg)
    if model_path.exists():
        return str(model_path.resolve())
    return model_path_arg


@dataclass(frozen=True)
class PackedRow:
    sample_id: str
    task_type: str
    text: str
    response_start_char: int


class PackedConversationDataset(Dataset):
    def __init__(
        self,
        *,
        parquet_path: Path,
        tokenizer: Any,
        max_length: int,
        enable_packing: bool,
        packing_max_length: int,
        packing_max_samples_per_pack: int,
        packing_cache_dir: Path | None,
    ) -> None:
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.enable_packing = bool(enable_packing)
        self.packing_max_length = min(int(packing_max_length), self.max_length)
        self.packing_max_samples_per_pack = int(packing_max_samples_per_pack)
        self.split = infer_split_from_paths([str(parquet_path)])
        self.records = self._load_records()
        self._tokenized_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.skipped_indices: set[int] = set()
        self.pack_groups: list[PackGroup] = []
        self.summary: dict[str, Any] = {}
        self.cache_path = self._build_cache_path(packing_cache_dir)
        self._prepare_groups()

    def __len__(self) -> int:
        return len(self.pack_groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pack_group = self.pack_groups[index]
        sample_tensors = [self._tokenize_record(sample_index) for sample_index in pack_group.sample_indices]
        input_ids = torch.cat([sample["input_ids"] for sample in sample_tensors], dim=0)
        completion_mask = torch.cat([sample["completion_mask"] for sample in sample_tensors], dim=0)
        seq_lengths = [int(sample["input_ids"].shape[0]) for sample in sample_tensors]
        return {
            "input_ids": input_ids.tolist(),
            "completion_mask": completion_mask.tolist(),
            "seq_lengths": seq_lengths,
        }

    def _load_records(self) -> list[PackedRow]:
        table = pq.read_table(
            self.parquet_path,
            columns=["id", "task_type", "text", "response_start_char"],
        )
        ids = table.column("id").to_pylist()
        task_types = table.column("task_type").to_pylist()
        texts = table.column("text").to_pylist()
        response_starts = table.column("response_start_char").to_pylist()
        records = [
            PackedRow(
                sample_id=str(sample_id),
                task_type=str(task_type),
                text=str(text),
                response_start_char=int(response_start_char),
            )
            for sample_id, task_type, text, response_start_char in zip(ids, task_types, texts, response_starts)
        ]
        self._log(f"loaded {len(records)} rendered rows from {self.parquet_path.name}")
        return records

    def _tokenize_record(self, index: int) -> dict[str, torch.Tensor]:
        cached = self._tokenized_cache.get(index)
        if cached is not None:
            return cached

        record = self.records[index]
        text_tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        encoded = text_tokenizer(
            record.text,
            add_special_tokens=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        offsets = encoded["offset_mapping"]
        completion_mask = torch.tensor(
            [1 if int(end) > record.response_start_char else 0 for _, end in offsets],
            dtype=torch.long,
        )
        result = {
            "input_ids": input_ids,
            "completion_mask": completion_mask,
        }
        self._tokenized_cache[index] = result
        return result

    def _prepare_groups(self) -> None:
        cached = self._maybe_load_cached_plan()
        if cached is not None:
            self.pack_groups, self.summary, skipped = cached
            self.skipped_indices = set(skipped)
            overall = self.summary["overall"]
            self._log(
                "loaded cached packing plan: "
                f"kept={overall['kept_samples']} skipped={overall['skipped_too_long_samples']} "
                f"packed_items={overall['packed_items']}"
            )
            return

        packable_items_by_task: dict[str, list[tuple[int, int]]] = defaultdict(list)
        passthrough_groups: list[PackGroup] = []
        raw_counts_by_task: dict[str, int] = defaultdict(int)
        skipped_counts_by_task: dict[str, int] = defaultdict(int)

        for sample_index, record in enumerate(self.records):
            raw_counts_by_task[record.task_type] += 1
            sample = self._tokenize_record(sample_index)
            sample_length = int(sample["input_ids"].shape[0])
            if sample_length > self.max_length:
                self.skipped_indices.add(sample_index)
                skipped_counts_by_task[record.task_type] += 1
                continue

            if self.enable_packing and record.task_type in SHORT_TASK_TYPES:
                packable_items_by_task[record.task_type].append((sample_index, sample_length))
            else:
                passthrough_groups.append(
                    PackGroup(
                        task_type=record.task_type,
                        sample_indices=(sample_index,),
                        total_length=sample_length,
                    )
                )

            if (sample_index + 1) % 2000 == 0:
                self._log(
                    f"packing scan {sample_index + 1}/{len(self.records)} | "
                    f"eligible={sum(len(items) for items in packable_items_by_task.values())} "
                    f"skipped={len(self.skipped_indices)}"
                )

        packed_groups: list[PackGroup] = []
        groups_by_task: dict[str, list[PackGroup]] = {}
        for task_type in SHORT_TASK_TYPES:
            task_items = packable_items_by_task.get(task_type, [])
            if not task_items:
                continue
            task_groups = first_fit_decreasing_pack(
                task_type=task_type,
                items=task_items,
                max_length=self.packing_max_length,
                max_samples_per_pack=self.packing_max_samples_per_pack,
            )
            packed_groups.extend(task_groups)
            groups_by_task[task_type] = task_groups

        self.pack_groups = sort_groups_by_first_index(passthrough_groups + packed_groups)
        packing_summary = build_short_task_packing_summary(
            groups_by_task=groups_by_task,
            raw_counts_by_task={
                task_type: raw_counts_by_task[task_type] - skipped_counts_by_task.get(task_type, 0)
                for task_type in SHORT_TASK_TYPES
                if raw_counts_by_task.get(task_type, 0) - skipped_counts_by_task.get(task_type, 0) > 0
            },
            max_length=self.packing_max_length,
            too_long_counts_by_task={task: skipped_counts_by_task.get(task, 0) for task in SHORT_TASK_TYPES},
        )

        total_raw = len(self.records)
        total_kept = total_raw - len(self.skipped_indices)
        non_packing_kept = sum(1 for group in passthrough_groups)
        self.summary = {
            "split": self.split,
            "max_length": self.max_length,
            "packing_enabled": self.enable_packing,
            "packing_max_length": self.packing_max_length,
            "packing_max_samples_per_pack": self.packing_max_samples_per_pack,
            "overall": {
                "raw_samples": total_raw,
                "kept_samples": total_kept,
                "skipped_too_long_samples": len(self.skipped_indices),
                "packed_items": len(self.pack_groups),
                "packing_tasks_kept_samples": packing_summary["raw_samples"],
                "packing_tasks_packed_items": packing_summary["packed_items"],
                "packing_tasks_saved_items": packing_summary["saved_items"],
                "packing_tasks_multi_sample_packs": packing_summary["multi_sample_packs"],
                "non_packing_singletons": non_packing_kept,
            },
            "packing_tasks": packing_summary["tasks"],
            "skipped_too_long_by_task": dict(sorted(skipped_counts_by_task.items())),
        }
        self._write_cached_plan()
        overall = self.summary["overall"]
        self._log(
            "prepared packing plan: "
            f"kept={overall['kept_samples']} skipped={overall['skipped_too_long_samples']} "
            f"packed_items={overall['packed_items']}"
        )

    def _build_cache_path(self, packing_cache_dir: Path | None) -> Path:
        cache_dir = packing_cache_dir or self.parquet_path.resolve().parent
        fingerprint_payload = json.dumps(
            {
                "parquet_path": str(self.parquet_path.resolve()),
                "max_length": self.max_length,
                "packing_enabled": self.enable_packing,
                "packing_max_length": self.packing_max_length,
                "packing_max_samples_per_pack": self.packing_max_samples_per_pack,
                "rows": len(self.records),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        fingerprint = hashlib.sha1(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
        return cache_dir / f"{self.parquet_path.stem}.unsloth_packing_{fingerprint}.json"

    def _maybe_load_cached_plan(self) -> tuple[list[PackGroup], dict[str, Any], list[int]] | None:
        if self.cache_path.exists():
            return self._load_cached_plan(self.cache_path)
        if not is_main_process():
            deadline = time.time() + PACKING_CACHE_WAIT_SECONDS
            while time.time() < deadline:
                if self.cache_path.exists():
                    return self._load_cached_plan(self.cache_path)
                time.sleep(2.0)
            self._log(f"packing cache wait timed out: {self.cache_path}")
        return None

    def _write_cached_plan(self) -> None:
        payload = {
            "summary": self.summary,
            "skipped_indices": sorted(self.skipped_indices),
            "groups": [
                {
                    "task_type": group.task_type,
                    "sample_indices": list(group.sample_indices),
                    "total_length": group.total_length,
                }
                for group in self.pack_groups
            ],
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.cache_path)
        self._log(f"saved packing cache: {self.cache_path}")

    @staticmethod
    def _load_cached_plan(cache_path: Path) -> tuple[list[PackGroup], dict[str, Any], list[int]]:
        with cache_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        groups = [
            PackGroup(
                task_type=str(group["task_type"]),
                sample_indices=tuple(int(index) for index in group["sample_indices"]),
                total_length=int(group["total_length"]),
            )
            for group in payload["groups"]
        ]
        return groups, dict(payload["summary"]), [int(index) for index in payload.get("skipped_indices", [])]

    @staticmethod
    def _log(message: str) -> None:
        if is_main_process():
            print(f"[packing] {message}", flush=True)


def rendered_cache_path(raw_parquet: Path, cache_dir: Path, model_ref: str, enable_thinking: bool) -> Path:
    stat = raw_parquet.stat()
    payload = json.dumps(
        {
            "path": str(raw_parquet.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "model_ref": model_ref,
            "enable_thinking": enable_thinking,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    fingerprint = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{raw_parquet.stem}.unsloth_rendered_{fingerprint}.parquet"


def ensure_rendered_parquet(
    raw_parquet: Path,
    cache_dir: Path,
    model_ref: str,
    trust_remote_code: bool,
    enable_thinking: bool,
    local_files_only: bool,
) -> Path:
    output_path = rendered_cache_path(raw_parquet, cache_dir, model_ref, enable_thinking)
    if output_path.exists():
        return output_path

    if is_main_process():
        render_table(
            input_path=raw_parquet,
            output_path=output_path,
            model_path=Path(model_ref),
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            local_files_only=local_files_only,
        )
        return output_path

    deadline = time.time() + RENDER_WAIT_SECONDS
    while time.time() < deadline:
        if output_path.exists():
            return output_path
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for rendered parquet: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unsloth multi-GPU SFT with precomputed no-truncation packing.")
    parser.add_argument("--data-dir", default="data/verl_storyboard_sft")
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--val-parquet", default=None)
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", default="outputs/verl_storyboard_sft")
    parser.add_argument("--project-name", default="storyboard-verl-sft")
    parser.add_argument("--experiment-name", default="qwen3-8b-storyboard-unsloth")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="lora")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--save-freq", type=int, default=40)
    parser.add_argument("--eval-freq", type=int, default=40)
    parser.add_argument("--logger", default="none")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--use-remove-padding", action="store_true", default=True)
    parser.add_argument("--no-use-remove-padding", dest="use_remove_padding", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use-short-task-packing", action="store_true", default=True)
    parser.add_argument("--no-use-short-task-packing", dest="use_short_task_packing", action="store_false")
    parser.add_argument("--use-eval-packing", action="store_true", default=True)
    parser.add_argument("--no-use-eval-packing", dest="use_eval_packing", action="store_false")
    parser.add_argument("--packing-max-length", type=int, default=8192)
    parser.add_argument("--packing-max-samples-per-pack", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--load-in-8bit", action="store_true", default=False)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--no-local-files-only", dest="local_files_only", action="store_false")
    parser.add_argument("--rendered-cache-dir", default=None)
    parser.add_argument("--packing-cache-dir", default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--resume-from-checkpoint", default=None)
    # Keep legacy flags for CLI compatibility, but early stopping is disabled.
    parser.add_argument("--early-stopping-patience", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def maybe_relaunch(args: argparse.Namespace) -> bool:
    world_size = args.nproc_per_node * args.nnodes
    if world_size <= 1 or args._worker or "LOCAL_RANK" in os.environ:
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
        "--_worker",
        *[arg for arg in sys.argv[1:] if arg != "--_worker"],
    ]
    print("[unsloth] launch command", flush=True)
    print(quote_for_display(cmd), flush=True)
    if args.dry_run:
        return True
    subprocess.run(cmd, check=True)
    return True


def build_dataset(tokenizer: Any, parquet_path: Path, args: argparse.Namespace, enable_packing: bool) -> PackedConversationDataset:
    packing_cache_dir = Path(args.packing_cache_dir) if args.packing_cache_dir else None
    return PackedConversationDataset(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        enable_packing=enable_packing,
        packing_max_length=min(args.packing_max_length, args.max_length),
        packing_max_samples_per_pack=args.packing_max_samples_per_pack,
        packing_cache_dir=packing_cache_dir,
    )


def resolve_text_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def train(args: argparse.Namespace) -> None:
    if not check_module("unsloth"):
        raise RuntimeError("当前 Python 环境未安装 unsloth，先安装后再启动训练")
    if not check_module("trl"):
        raise RuntimeError("当前 Python 环境未安装 trl，先安装后再启动训练")

    from trl import SFTConfig, SFTTrainer
    try:
        from trl import DataCollatorForLanguageModeling
    except ImportError:
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from unsloth import FastLanguageModel
    from unsloth.utils.packing import enable_padding_free_metadata

    data_dir = Path(args.data_dir)
    train_parquet = Path(args.train_parquet) if args.train_parquet else data_dir / "train.parquet"
    val_parquet = Path(args.val_parquet) if args.val_parquet else data_dir / "val.parquet"
    output_dir = Path(args.output_dir)
    rendered_cache_dir = Path(args.rendered_cache_dir) if args.rendered_cache_dir else data_dir / "unsloth_rendered"
    model_ref = resolve_model_ref(args.model_path)

    if not train_parquet.exists():
        raise FileNotFoundError(f"训练集不存在: {train_parquet}")
    if not val_parquet.exists():
        raise FileNotFoundError(f"验证集不存在: {val_parquet}")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("--load-in-4bit 和 --load-in-8bit 只能二选一")
    if args.finetune_mode == "full" and (args.load_in_4bit or args.load_in_8bit):
        raise ValueError("full finetuning 不应与 4bit/8bit 量化加载同时使用")
    if args.use_short_task_packing and not args.use_remove_padding:
        raise ValueError("自定义 packing 依赖 remove padding；关闭后会发生样本边界串扰，请开启 --use-remove-padding")
    rendered_train = ensure_rendered_parquet(
        raw_parquet=train_parquet,
        cache_dir=rendered_cache_dir,
        model_ref=model_ref,
        trust_remote_code=args.trust_remote_code,
        enable_thinking=args.enable_thinking,
        local_files_only=args.local_files_only,
    )
    rendered_val = ensure_rendered_parquet(
        raw_parquet=val_parquet,
        cache_dir=rendered_cache_dir,
        model_ref=model_ref,
        trust_remote_code=args.trust_remote_code,
        enable_thinking=args.enable_thinking,
        local_files_only=args.local_files_only,
    )
    maybe_barrier()

    # In DDP, each rank must load the model on its own GPU to avoid
    # accelerate device-placement errors with quantized/16-bit models.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model, tokenizer = FastLanguageModel.from_pretrained(
        **supported_kwargs(
            FastLanguageModel.from_pretrained,
            {
                "model_name": model_ref,
                "max_seq_length": args.max_length,
                "dtype": torch.bfloat16 if args.dtype == "bf16" else torch.float16,
                "load_in_4bit": args.load_in_4bit,
                "load_in_8bit": args.load_in_8bit,
                "load_in_16bit": not args.load_in_4bit and not args.load_in_8bit and args.finetune_mode == "lora",
                "device_map": {"": torch.cuda.current_device()},
                "full_finetuning": args.finetune_mode == "full",
                "trust_remote_code": args.trust_remote_code,
                "local_files_only": args.local_files_only,
                "use_gradient_checkpointing": "unsloth" if args.gradient_checkpointing else False,
            },
        )
    )
    # Keep the official Unsloth loading flow: reuse the tokenizer returned by
    # FastLanguageModel.from_pretrained instead of hard-coding a Qwen2Tokenizer.
    text_tokenizer = resolve_text_tokenizer(tokenizer)
    if text_tokenizer.eos_token != "<|im_end|>":
        text_tokenizer.eos_token = "<|im_end|>"
    if text_tokenizer.pad_token != "<|endoftext|>":
        text_tokenizer.pad_token = "<|endoftext|>"

    if args.finetune_mode == "lora":
        model = FastLanguageModel.get_peft_model(
            **supported_kwargs(
                FastLanguageModel.get_peft_model,
                {
                    "model": model,
                    "r": args.lora_rank,
                    "target_modules": OFFICIAL_LORA_TARGET_MODULES,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "bias": "none",
                    "use_gradient_checkpointing": "unsloth" if args.gradient_checkpointing else False,
                    "use_rslora": False,
                    "loftq_config": None,
                    "max_seq_length": args.max_length,
                },
            )
        )

    train_dataset = build_dataset(text_tokenizer, rendered_train, args, enable_packing=args.use_short_task_packing)
    eval_dataset = build_dataset(text_tokenizer, rendered_val, args, enable_packing=args.use_eval_packing)

    collator = DataCollatorForLanguageModeling(
        **supported_kwargs(
            DataCollatorForLanguageModeling,
            {
                "pad_token_id": text_tokenizer.pad_token_id,
                "completion_only_loss": True,
                "padding_free": args.use_remove_padding,
                "return_tensors": "pt",
            },
        )
    )

    report_to: list[str] | str = ["wandb"] if "wandb" in args.logger.split(",") else "none"
    if report_to != "none":
        os.environ.setdefault("WANDB_PROJECT", args.project_name)
    if is_main_process():
        print(
            "[unsloth] text tokenizer: "
            f"{text_tokenizer.__class__.__name__} eos={text_tokenizer.eos_token!r} "
            f"eos_id={text_tokenizer.eos_token_id} pad={text_tokenizer.pad_token!r} "
            f"pad_id={text_tokenizer.pad_token_id}",
            flush=True,
        )

    sft_config_kwargs = {
        "output_dir": str(output_dir.resolve()),
        "run_name": args.experiment_name,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.micro_batch_size,
        "per_device_eval_batch_size": args.val_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": resolve_default_lr(args.finetune_mode, args.lr),
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.clip_grad,
        "logging_steps": 1,
        "save_steps": args.save_freq,
        "eval_steps": args.eval_freq,
        "save_strategy": "steps" if args.save_freq > 0 else "epoch",
        "eval_strategy": "steps" if args.eval_freq > 0 else "no",
        "report_to": report_to,
        "bf16": args.dtype == "bf16",
        "fp16": args.dtype == "fp16",
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "packing": False,
        "prediction_loss_only": True,
        "max_length": args.max_length,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "save_total_limit": args.save_total_limit,
        "ddp_find_unused_parameters": False,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }
    if args.warmup_ratio is None:
        sft_config_kwargs["warmup_steps"] = args.warmup_steps
    else:
        sft_config_kwargs["warmup_ratio"] = args.warmup_ratio
    callbacks: list[TrainerCallback] = [ClearCacheCallback()]

    trainer = SFTTrainer(
        **supported_kwargs(
            SFTTrainer,
            {
                "model": model,
                "args": SFTConfig(**supported_kwargs(SFTConfig, sft_config_kwargs)),
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset if args.eval_freq > 0 else None,
                "data_collator": collator,
                "processing_class": text_tokenizer,
                "tokenizer": text_tokenizer,
                "callbacks": callbacks,
            },
        )
    )
    if args.use_remove_padding:
        # Custom packed rows already carry `seq_lengths`; this injects the
        # metadata Unsloth needs to keep packed sequence boundaries isolated.
        enable_padding_free_metadata(model, trainer)

    if is_main_process():
        world_size = args.nproc_per_node * args.nnodes
        print(
            "[unsloth] batch config: "
            f"micro_batch={args.micro_batch_size}, grad_accum={args.grad_accum_steps}, "
            f"world_size={world_size}, train_batch_size={args.micro_batch_size * args.grad_accum_steps * world_size}",
            flush=True,
        )
        print(
            "[unsloth] packing config: "
            f"enabled={args.use_short_task_packing} max_length={args.max_length} "
            f"packing_max_length={min(args.packing_max_length, args.max_length)} "
            f"packing_max_samples_per_pack={args.packing_max_samples_per_pack}",
            flush=True,
        )
        if args.early_stopping_patience > 0 or args.early_stopping_threshold > 0:
            print("[unsloth] early stopping is disabled; legacy CLI flags are ignored", flush=True)
        if args.use_remove_padding:
            print("[unsloth] padding-free metadata enabled for custom packed boundaries", flush=True)
        print(f"[unsloth] train packing summary: {json.dumps(train_dataset.summary, ensure_ascii=False)}", flush=True)
        print(f"[unsloth] eval packing summary: {json.dumps(eval_dataset.summary, ensure_ascii=False)}", flush=True)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    maybe_barrier()
    trainer.save_model()
    if is_main_process():
        tokenizer.save_pretrained(output_dir)
        print(f"[unsloth] model saved to {output_dir.resolve()}", flush=True)



def main() -> None:
    args = build_parser().parse_args()
    if maybe_relaunch(args):
        return
    if args.dry_run:
        return
    train(args)


if __name__ == "__main__":
    main()
