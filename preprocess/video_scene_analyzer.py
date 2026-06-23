# 输入：视频文件夹。输出：saved/<子文件夹名>/plot/<视频名>.json

import os
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import dashscope
from dashscope import MultiModalConversation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_dashscope_api_key
from project_paths import DEFAULT_SAVED_DIR

# ==========================================
# 1. 加载配置与初始化
# ==========================================
API_KEY = get_dashscope_api_key()
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
if not API_KEY:
    raise RuntimeError("DashScope API key is required. Please set DASHSCOPE_API_KEY in config.py.")
dashscope.api_key = API_KEY

MODEL_NAME = "qwen3.5-plus"
MAX_WORKERS = 40  # 并发线程数

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.ts', '.m4v'}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = str(DEFAULT_SAVED_DIR)

# ==========================================
# 2. 工具函数
# ==========================================
def extract_json(text):
    if not text:
        return {}
    try:
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        print(f"       JSON解析失败，大模型原始输出为:\n{text}")
        return {}


def find_videos(input_folder):
    """扫描 input_folder 的直接子文件夹，收集所有视频文件。
    返回列表：[(视频绝对路径, 子文件夹名称, 集数索引), ...]
    索引与 extract_frames.py 一致：子文件夹内按文件名排序，从 1 开始。
    """
    videos = []
    for entry in sorted(os.scandir(input_folder), key=lambda e: e.name):
        if entry.is_dir():
            subfolder_name = entry.name
            files = sorted(
                (f for f in os.scandir(entry.path)
                 if f.is_file() and os.path.splitext(f.name)[1].lower() in VIDEO_EXTENSIONS),
                key=lambda f: int(os.path.splitext(f.name)[0]) if os.path.splitext(f.name)[0].isdigit() else f.name,
            )
            for idx, file in enumerate(files, start=1):
                stem = os.path.splitext(file.name)[0]
                idx = int(stem) if stem.isdigit() else idx
                videos.append((file.path, subfolder_name, idx))
    return videos


def get_output_path(subfolder_name, index):
    out_dir = os.path.join(SAVED_DIR, subfolder_name, "plot")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{index}.json")


# ==========================================
# 3. LLM 调用（带重试）
# ==========================================
def analyze_video(video_path, max_retries=3):
    prompt = """你是一个专业的 AI 剧情分析师和【潜台词（Subtext）推理大师】。
    我为你提供了一段【动态视频】，请综合分析视频内容，输出人物表、剧情梗概和全局对话。

    ⚠️【语言与格式要求】⚠️：
    1. 所有文本内容必须【全部使用简体中文】！绝对禁止输出任何英文单词！
    2. 必须直接输出纯 JSON 数据，严禁输出任何分析过程、道歉或解释性废话！

    【提取内容】
    1. characters：识别所有人物，提供中文名称和客观外貌描述。无法辨认时返回 []。
    2. plot_summary：总结核心事件，必须识破人物间的潜台词与真实氛围（如冷嘲热讽、针锋相对、隐忍等），拒绝流水账式字面总结。
    3. dialogues：提取对话。
       - **防遗漏**：所有真实台词都必须包含，不得遗漏。
       - **防捏造**：没有台词必须返回 []，严禁脑补台词。
       - **口型校验**：嘴唇紧闭或仅有情绪反应时，该声音是画外音，需追溯真正说话者。
       - **间接发声**：当画面人物处于接听电话、播放语音消息、阅读短信等状态时，实际说话人可能在画面之外。需根据台词和画面独立判断发声者身份，为未出镜的配角单独命名，不得将其台词归并至画面人物名下，亦不得遗漏。

    请严格输出 JSON 格式（全部简体中文）：
    {
    "characters": [{"name": "中文名称", "features": "中文外貌描述..."}],
    "plot_summary": "中文剧情梗概...",
    "dialogues": ["XXX: ...", "XXX: ..."]
    }"""

    abs_path = os.path.abspath(video_path).replace("\\", "/")
    file_url = f"file://{abs_path}"

    for attempt in range(max_retries):
        if attempt == 0:
            print(f"  [分析中] {os.path.basename(video_path)}")
        else:
            print(f"  [重试 {attempt + 1}/{max_retries}] {os.path.basename(video_path)}")
            time.sleep(1)

        try:
            response = MultiModalConversation.call(
                api_key=API_KEY,
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"video": file_url, "fps": 2},
                        {"text": prompt}
                    ]
                }]
            )
            if response.status_code != 200:  # type: ignore[union-attr]
                print(f"  [失败] {os.path.basename(video_path)}: {response.message}")  # type: ignore[union-attr]
                continue

            raw_text = response.output.choices[0].message.content[0]["text"]
            result = extract_json(raw_text)
            if result.get("plot_summary") or result.get("dialogues") or result.get("characters"):
                return result
        except Exception as e:
            print(f"  [异常] {os.path.basename(video_path)}: {e}")

    print(f"  [放弃] {os.path.basename(video_path)}：达到最大重试次数 ({max_retries}次)")
    return {}


# ==========================================
# 4. 单视频处理任务
# ==========================================
def is_valid_result(output_path):
    """检查已保存的 JSON 是否包含有效内容（非空结果）。"""
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("plot_summary") or data.get("dialogues") or data.get("characters"))
    except Exception:
        return False


def process_video(video_path, subfolder_name, index):
    output_path = get_output_path(subfolder_name, index)
    if os.path.exists(output_path):
        if is_valid_result(output_path):
            print(f"  [跳过] 已存在: {output_path}")
            return output_path, True
        else:
            print(f"  [重新分析] 上次结果为空: {os.path.basename(video_path)}")

    result = analyze_video(video_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"  [完成] {output_path}")
    return output_path, bool(result)


# ==========================================
# 5. 主流程
# ==========================================
def main():
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "videos"

    if not os.path.isdir(input_folder):
        print(f"找不到文件夹: {input_folder}")
        sys.exit(1)

    videos = find_videos(input_folder)
    if not videos:
        print(f"在 {input_folder} 的子文件夹中未找到视频文件")
        sys.exit(0)

    print(f"找到 {len(videos)} 个视频，使用 {MAX_WORKERS} 线程并发分析...\n")

    success, failed = 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_video, vp, sn, idx): (vp, sn)
            for vp, sn, idx in videos
        }
        for future in as_completed(futures):
            vp, _ = futures[future]
            try:
                _, ok = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  [错误] {vp}: {e}")
                failed += 1

    print(f"\n全部完成：成功 {success}，失败 {failed} 个")
    print(f"结果保存在: {SAVED_DIR}")


if __name__ == "__main__":
    main()
