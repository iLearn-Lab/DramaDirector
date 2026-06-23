"""
infer.py — 从 checkpoint 加载 LoRA 模型，抽取 test set 中若干样本进行推理，
           并对比 ground truth 输出。

用法：
    python infer.py --checkpoint outputs/checkpoint-XXX
    python infer.py --checkpoint qwen_storyboard_sft_lora   # 最终保存的 LoRA
    python infer.py --checkpoint outputs/checkpoint-XXX --base_model ./Qwen/Qwen3.5-9B
"""

import argparse
import json

from unsloth import FastLanguageModel
from transformers import TextStreamer

# ============================================================================
# 与训练脚本完全一致的 System Prompt
# ============================================================================
END_TOKEN = "<END>"

_SHOT_TEMPLATE = '''\
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
]'''

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
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--output_file", type=str, default="infer_results.json", help="推理结果保存路径")
    return parser.parse_args()


def load_model(checkpoint: str, base_model: str):
    """加载 base 模型后合并 LoRA checkpoint 权重，切换到推理模式。"""
    print(f"Loading base model from: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        use_gradient_checkpointing=False,
        max_seq_length=32768,
        local_files_only=True,
    )

    print(f"Loading LoRA weights from: {checkpoint}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint)

    FastLanguageModel.for_inference(model)
    print("Model ready.\n")
    return model, tokenizer


def run_inference(model, tokenizer, user_content: str, args) -> str:
    """执行单条推理，流式输出结果，并返回生成文本。"""
    import logging
    logging.getLogger("transformers.processing_utils").setLevel(logging.ERROR)
    # 文本模型的 chat template 期望 content 为字符串。
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
    except TypeError:
        # 一些 tokenizer 版本不支持 enable_thinking 参数
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

    _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    im_end_id = _tok.convert_tokens_to_ids("<|im_end|>")
    base_eos = _tok.eos_token_id
    eos_ids = list({base_eos, im_end_id} - {None})

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    output_ids = model.generate(
        **inputs.to(model.device),
        streamer=streamer,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=eos_ids,
    )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    args = parse_args()

    # 加载测试集
    print(f"Loading test data from: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    samples = test_data[: min(args.num_samples, len(test_data))]
    print(f"Selected {len(samples)} samples\n")

    # 加载模型
    model, tokenizer = load_model(args.checkpoint, args.base_model)

    # 逐条推理
    results = []
    for i, sample in enumerate(samples, 1):
        print("=" * 80)
        print(f"[Sample {i}/{len(samples)}]")
        print("-" * 40)
        print("[USER PROMPT]")
        print(sample["user"])
        print("-" * 40)
        print("[GROUND TRUTH]")
        print(sample["assistant"])
        print("-" * 40)
        print("[MODEL OUTPUT]")
        model_output = run_inference(model, tokenizer, sample["user"], args)
        print()
        results.append({
            "sample_index": i,
            "user": sample["user"],
            "ground_truth": sample["assistant"],
            "model_output": model_output,
        })

    print("=" * 80)
    print("Done.")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
