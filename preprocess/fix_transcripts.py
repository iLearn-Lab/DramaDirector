# 输入：递归查找文件夹中的 transcript/*.json
# SCENE_JSON：对应 plot/<N>.json
# 输出：transcript/fixed_<N>.json

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_dashscope_api_key

# ==========================================
# 1. 配置
# ==========================================
API_KEY = get_dashscope_api_key()
MODEL_NAME = "qwen3.5-plus"
MAX_WORKERS = 40

if not API_KEY:
    raise RuntimeError("DashScope API key is required. Please set DASHSCOPE_API_KEY in config.py.")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ==========================================
# 2. 工具函数
# ==========================================
def extract_json(text):
    if not text:
        return {}
    try:
        stripped = re.sub(r'^```[^\n]*\n?', '', text.strip(), flags=re.MULTILINE).strip().rstrip('`').strip()
        try:
            return json.loads(stripped)
        except Exception:
            pass
        for match in reversed(list(re.finditer(r'\{', text))):
            candidate = text[match.start():]
            end = candidate.rfind('}')
            if end != -1:
                try:
                    return json.loads(candidate[:end + 1])
                except Exception:
                    continue
        return json.loads(text)
    except Exception:
        print(f"  JSON解析失败，大模型原始输出为:\n{text}")
        return {}


def fix_transcripts(transcripts: dict, scene_data: dict, expected_count: int | None = None, max_retries=3) -> dict:
    characters = scene_data.get("characters", [])
    char_lines = "\n".join(
        f"  - {c['name']}：{c['features']}" for c in characters
    ) or "  （无）"

    plot = scene_data.get("plot_summary", "（无）")

    dialogue_lines = "\n".join(
        f"  {d}" for d in scene_data.get("dialogues", [])
    ) or "  （无）"

    output_skeleton = {k: "..." for k in transcripts.keys()}
    output_format_str = json.dumps(output_skeleton, ensure_ascii=False, indent=2)
    transcripts_str = json.dumps(transcripts, ensure_ascii=False, indent=2)

    prompt = (
        "你是一位专业的影视台词校对师。以下是一集短剧的相关信息：\n\n"
        "【人物表】\n"
        f"{char_lines}\n\n"
        "【剧情梗概】\n"
        f"  {plot}\n\n"
        "【台词参考（可能内容不全，但是内容正确）】\n"
        f"{dialogue_lines}\n\n"
        "【待修复的分镜台词（ASR语音识别结果，可能含错误）】\n"
        f"{transcripts_str}\n\n"
        "【修复规则】\n"
        "1. 对照【台词参考】，修正分镜台词中的音近字、人名错误、语义不通等识别错误。\n"
        "2. 空字符串的分镜保持为空字符串，不要填入任何内容。\n"
        "3. 保持每条台词的简洁性，不要扩写或添加原文没有的内容。\n"
        "4. 分镜编号（JSON 的 key）保持不变，数量与顺序必须与输入完全一致。\n\n"
        "【输出格式硬性要求】\n"
        "- 直接输出修复后的 JSON 对象，严禁输出任何解释、前言或 markdown 代码块。\n"
        "- 严格按照以下骨架格式输出：\n"
        f"{output_format_str}\n"
    )

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"  解析失败，第 {attempt + 1}/{max_retries} 次重试...")
            time.sleep(1)
        current_prompt = prompt
        if attempt > 0 and expected_count is not None:
            current_prompt += f"\n\n⚠️ 再次提醒：输出的JSON必须恰好包含 {expected_count} 个key，最大index为 {expected_count - 1}。"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": current_prompt}]
            )
            raw_text = response.choices[0].message.content
            result = extract_json(raw_text)
            if isinstance(result, dict) and len(result) > 0:
                if expected_count is None or len(result) == expected_count:
                    return result
                print(f"  分镜数量不匹配（期望 {expected_count}，实际 {len(result)}），重试...")
        except Exception as e:
            print(f"  调用失败: {e}")

    print(f"  达到最大重试次数 ({max_retries} 次)，修复失败。")
    return {}


# ==========================================
# 3. 查找任务
# ==========================================
def find_tasks(root: str):
    """
    递归查找所有 transcript/<N>.json，
    对应 plot/<N>.json，输出到 transcript/fixed_<N>.json
    返回 list of (transcript_path, plot_path, output_path)
    """
    tasks = []
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) != "transcript":
            continue
        show_dir = os.path.dirname(dirpath)
        plot_dir = os.path.join(show_dir, "plot")
        for fname in sorted(filenames):
            if not fname.endswith(".json"):
                continue
            if fname.startswith("fixed_"):
                continue
            index = fname[:-5]  # 去掉 .json
            plot_path = os.path.join(plot_dir, fname)
            if not os.path.exists(plot_path):
                print(f"  [跳过] 缺少对应 plot 文件: {plot_path}")
                continue
            transcript_path = os.path.join(dirpath, fname)
            output_path = os.path.join(dirpath, f"fixed_{index}.json")
            tasks.append((transcript_path, plot_path, output_path, index))
    return tasks


# ==========================================
# 4. 处理单个任务
# ==========================================
def process_task(task):
    transcript_path, plot_path, output_path, index = task

    if os.path.exists(output_path):
        print(f"  [跳过] 已存在: {output_path}")
        return index, True

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)
    with open(plot_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    # 从 split 目录获取基准分镜数量
    show_dir = os.path.dirname(os.path.dirname(transcript_path))
    split_timeline = os.path.join(show_dir, "split", index, "segments_timeline.json")
    expected_count = None
    if os.path.exists(split_timeline):
        with open(split_timeline, "r", encoding="utf-8") as f:
            expected_count = len(json.load(f).get("segments", []))

    print(f"  [修复] 第 {index} 集，共 {len(transcripts)} 条分镜（期望 {expected_count}）...")
    fixed = fix_transcripts(transcripts, scene_data, expected_count=expected_count)

    if not fixed:
        print(f"  [失败] 第 {index} 集修复失败。")
        return index, False

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"  [完成] 第 {index} 集 → {output_path}")
    return index, True


# ==========================================
# 5. 主流程
# ==========================================
def main():
    if len(sys.argv) < 2:
        print("用法: python fix_transcripts.py <文件夹>")
        sys.exit(1)

    root = sys.argv[1]
    tasks = find_tasks(root)

    if not tasks:
        print("未找到任何待修复的 transcript 文件。")
        return

    print(f"找到 {len(tasks)} 个任务，使用 {MAX_WORKERS} 个并发线程...\n")

    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_task, t): t for t in tasks}
        for future in as_completed(futures):
            _, ok = future.result()
            if ok:
                success += 1
            else:
                fail += 1

    print(f"\n完成：{success} 成功，{fail} 失败。")


if __name__ == "__main__":
    main()
