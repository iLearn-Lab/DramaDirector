"""
infer_vllm.py — 使用 vLLM + LoRA 进行推理，
                抽取 test set 中若干样本并对比 ground truth。

用法：
    python infer_vllm.py --checkpoint outputs/checkpoint-XXX
    python infer_vllm.py --checkpoint qwen_storyboard_sft_lora
    python infer_vllm.py --checkpoint outputs/checkpoint-XXX --base_model ./Qwen/Qwen3.5-9B
"""

import argparse
import json
import os
import re
from typing import List

from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise ImportError(
        "未检测到 vLLM，请先安装：pip install vllm"
    ) from e

try:
    from vllm.lora.request import LoRARequest
except ImportError:
    # 兼容部分版本的导入路径
    from vllm import LoRARequest


# ============================================================================
# 与训练脚本一致的 System Prompt
# ============================================================================
END_TOKEN = "<END>"

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
    f"- 当本集所有分镜全部输出完毕后，在 JSON 数组末尾的右括号后另起一行输出 {END_TOKEN}"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="qwen_storyboard_sft_lora",
        help="LoRA checkpoint 目录（outputs/checkpoint-XXX 或最终保存目录）",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="./Qwen/Qwen3.5-9B",
        help="base 模型路径",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="generation/test.json",
        help="测试集 JSON 文件路径",
    )
    parser.add_argument("--num_samples", type=int, default=3, help="抽取样本数")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="每批送入 vLLM 的样本数；每批完成后会增量落盘",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--min_p", type=float)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--output_file", type=str, default="infer_results.json", help="推理结果保存路径"
    )

    # vLLM 相关参数
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument(
        "--disable_chunked_prefill",
        action="store_true",
        help="禁用 chunked prefill（部分模型/场景可提高稳定性）",
    )
    return parser.parse_args()


def build_prompt(tokenizer: AutoTokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # 一些 tokenizer 版本不支持 enable_thinking 参数
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_stop_token_ids(tokenizer: AutoTokenizer) -> List[int]:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    return list({tid for tid in [eos_id, im_end_id] if tid is not None})


def batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def save_results(results, output_file: str):
    tmp_output_file = f"{output_file}.tmp"
    with open(tmp_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_output_file, output_file)


def create_llm(args):
    llm_kwargs = {
        "model": args.base_model,
        "tokenizer": args.base_model,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enable_lora": True,
        "max_lora_rank": args.max_lora_rank,
        "disable_chunked_prefill": args.disable_chunked_prefill,
    }

    while True:
        try:
            return LLM(**llm_kwargs)
        except TypeError as exc:
            message = str(exc)
            match = re.search(r"unexpected keyword argument '([^']+)'", message)
            if not match:
                raise

            bad_key = match.group(1)
            if bad_key not in llm_kwargs:
                raise

            print(
                "Warning: current vLLM version does not support LLM arg "
                f"'{bad_key}', skipping it."
            )
            llm_kwargs.pop(bad_key)


def main():
    args = parse_args()

    print(f"Loading test data from: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    samples = test_data[: min(args.num_samples, len(test_data))]
    print(f"Selected {len(samples)} samples\n")

    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )

    stop_token_ids = get_stop_token_ids(tokenizer)
    print(f"Loading vLLM model from: {args.base_model}")
    llm = create_llm(args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    print(f"Loading LoRA weights from: {args.checkpoint}")
    lora_request = LoRARequest("storyboard-lora", 1, args.checkpoint)

    results = []
    total_samples = len(samples)
    batch_size = max(1, args.batch_size)

    print(
        f"Generating with vLLM in batches: total_samples={total_samples}, "
        f"batch_size={batch_size}\n"
    )

    for batch_start, batch_samples in batched(samples, batch_size):
        batch_end = batch_start + len(batch_samples)
        print(
            f"Running batch {batch_start // batch_size + 1}: "
            f"samples {batch_start + 1}-{batch_end}/{total_samples}"
        )
        batch_prompts = [
            build_prompt(tokenizer, sample["user"]) for sample in batch_samples
        ]
        batch_outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=True,
        )

        for offset, (sample, output) in enumerate(zip(batch_samples, batch_outputs), 1):
            sample_index = batch_start + offset
            model_output = output.outputs[0].text if output.outputs else ""

            print("=" * 80)
            print(f"[Sample {sample_index}/{total_samples}]")
            print("-" * 40)
            print("[USER PROMPT]")
            print(sample["user"])
            print("-" * 40)
            print("[GROUND TRUTH]")
            print(sample["assistant"])
            print("-" * 40)
            print("[MODEL OUTPUT]")
            print(model_output)
            print()

            results.append(
                {
                    "sample_index": sample_index,
                    "user": sample["user"],
                    "ground_truth": sample["assistant"],
                    "model_output": model_output,
                }
            )

        save_results(results, args.output_file)
        print(
            f"Checkpoint saved: {len(results)}/{total_samples} results -> "
            f"{args.output_file}"
        )

    print("=" * 80)
    print("Done.")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
