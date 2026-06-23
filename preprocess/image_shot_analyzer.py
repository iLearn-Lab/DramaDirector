# 输入：saved 文件夹，递归查找 split/<N>/ 中的 keyframes
# SCENE_JSON：plot/<N>.json；台词：transcript/fixed_<N>.json；时长：split/<N>/segments_timeline.json
# 输出：shot/<N>.json

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

from config import get_dashscope_api_keys

# ==========================================
# 1. 配置
# ==========================================
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

API_KEYS = get_dashscope_api_keys()
if not API_KEYS:
    raise RuntimeError("DashScope API key is required. Please set DASHSCOPE_API_KEY or DASHSCOPE_API_KEYS in config.py.")
MODEL_NAME = "qwen3.5-plus"
MAX_WORKERS = 24

# ==========================================
# 2. 工具函数
# ==========================================
def extract_json(text):
    if not text:
        return []
    try:
        stripped = re.sub(r'^```[^\n]*\n?', '', text.strip(), flags=re.MULTILINE).strip().rstrip('`').strip()
        try:
            return json.loads(stripped)
        except Exception:
            pass
        for match in reversed(list(re.finditer(r'\[', text))):
            candidate = text[match.start():]
            bracket_end = candidate.rfind(']')
            if bracket_end != -1:
                try:
                    return json.loads(candidate[:bracket_end + 1])
                except Exception:
                    continue
        for match in reversed(list(re.finditer(r'\{', text))):
            candidate = text[match.start():]
            bracket_end = candidate.rfind('}')
            if bracket_end != -1:
                try:
                    return json.loads(candidate[:bracket_end + 1])
                except Exception:
                    continue
        return json.loads(text)
    except Exception:
        print(f"  JSON解析失败，大模型原始输出为:\n{text}")
        return []


def load_scene_info(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    characters = data.get("characters", [])
    char_lines = "\n".join(
        "  - {}：{}".format(c.get("name", "未知"), c.get("features", "无描述")) for c in characters
    ) or "  （无）"
    plot = data.get("plot_summary", "（无）")
    dialogue_lines = "\n".join(f"  {d}" for d in data.get("dialogues", [])) or "  （无）"
    return char_lines, plot, dialogue_lines


def load_transcripts(json_path):
    """返回 {segment_index(int): text(str)}，文件不存在返回空字典。"""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_timeline(json_path):
    """返回 (segments_ordered, durations_dict)。
    segments_ordered: 按 segment_index 排序的 segment 列表
    durations_dict:   {segment_index(int): duration_sec(float)}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = sorted(data.get("segments", []), key=lambda s: s["segment_index"])
    durations = {
        seg["segment_index"]: round(seg["end_sec"] - seg["start_sec"], 2)
        for seg in segments
    }
    return segments, durations


def build_description_visual(shot: dict) -> str:
    """
    静态从结构化字段拼接 description_visual。
    不含任何人名，专为 depth/pose 图检索优化。
    描述内容：镜头参数 + 场景背景 + 空间层次 + 每个人物的服装/位置/动作/表情。
    """
    parts = []

    # 1. 镜头参数
    scale  = (shot.get("shot_scale")    or "").strip()
    angle  = (shot.get("camera_angle")  or "").strip()
    motion = (shot.get("camera_motion") or "固定").strip()
    cam_tokens = [t for t in [scale, angle] if t]
    if cam_tokens:
        cam_str = "".join(cam_tokens) + "镜头"
        if motion and motion != "固定":
            cam_str += f"（{motion}）"
        parts.append(cam_str)

    # 2. 场景/背景
    bg = (shot.get("background") or "").strip()
    if bg:
        parts.append(bg)

    # 3. 人物（用位置+服装标识，无人名）
    for subj in (shot.get("subjects") or []):
        position  = (subj.get("position")   or "").strip()
        clothing  = (subj.get("clothing")   or "").strip()
        action    = (subj.get("action")     or "").strip()
        expression= (subj.get("expression") or "").strip()

        label = (position + "人物") if position else "人物"
        subj_parts = []
        if clothing:
            subj_parts.append(f"身穿{clothing}")
        if action:
            subj_parts.append(action)
        if expression:
            subj_parts.append(f"表情：{expression}")
        if subj_parts:
            parts.append(f"{label}，{'，'.join(subj_parts)}")

    return "；".join(parts) + "。" if parts else ""


# ==========================================
# 3. 分析函数
# ==========================================
def analyze_all(image_paths, scene_json_path, transcripts, durations, api_key, expected_count=None, max_retries=3):
    file_urls = [f"file://{p}" for p in image_paths]
    n = len(image_paths)

    char_lines, plot, dialogue_lines = load_scene_info(scene_json_path)

    # 为每张图提取 ASR 台词和时长
    per_shot_data = []
    for seq, path in enumerate(image_paths):
        fname = os.path.basename(path)
        nums = re.findall(r'\d+', fname)
        idx = int(nums[-1]) if nums else seq
        text = transcripts.get(idx, "").strip()
        duration = durations.get(idx, 0.0)
        per_shot_data.append((seq, idx, text, duration))

    has_transcripts = bool(transcripts)
    asr_block = (
        '【台词填写规则】\n'
        '每张图片后紧跟的"[台词]"标注是该帧对应视频片段的台词，直接复用'
    ) if has_transcripts else ""

    # 新结构：每个 shot 包含结构化字段 + 两种文本描述
    # description_visual  : 供 depth/pose 检索，不含人名
    # description_narrative: 供视频生成模型，含人名和叙事语境
    # subjects 中包含 name/gender，用于人物追踪和叙事生成
    output_skeleton = [
        {
            "index": i,
            "shot_scale": "...",        # 景别：大特写/特写/近景/中近景/中景/全景/远景
            "camera_angle": "...",      # 拍摄角度：俯拍/仰拍/平拍/低角度/斜角/过肩
            "camera_motion": "...",     # 镜头运动：固定/推进/拉远/摇镜/跟拍/升降
            "subjects": [               # 画面中每个人物一项
                {
                    "name":     "...",  # 人物姓名（与人物表一致；无法识别则填"未知"）
                    "gender":   "...",  # 性别：男/女/未知
                    "clothing": "...",  # 服装颜色+款式，作为人物标识
                    "position": "...",  # 在画面中的位置：画面左侧/右侧/中央/前景/背景
                    "action":   "...",  # 具体肢体动作+力度+方向
                    "expression": "..." # 眉/眼/口三个部位的微表情
                }
            ],
            "background": "...",        # 场景类型+关键道具+光线/氛围，一句话
            "description_narrative": "...",  # 含人名的叙事性描述，供视频生成用
            "dialogue": None,           # 台词原文（无则 null）
            "speaker": None,            # 说话人姓名（无则 null）
            "emotion": None,            # 说话人情绪（无则 null）
            "duration": 0.0
        }
        for i in range(n)
    ]
    output_format_str = json.dumps(output_skeleton[:2], ensure_ascii=False, indent=2)

    prompt = (
        f'你是一位专业的影视分镜师。我将为你提供连续的 {n} 张视频帧图片（每张图后附有该帧的 ASR 台词标注），'
        f'以及该片段的全局信息（人物表、剧情梗概、台词），请基于这些内容按顺序为每一帧生成结构化分镜数据。\n\n'
        '【全局信息】\n'
        f'人物表：\n{char_lines}\n\n'
        f'剧情梗概：\n  {plot}\n\n'
        + asr_block +
        '【总体要求】\n'
        '客观、精确、细节丰富，强化微动作与微表情，充满动态感，绝不主观臆断。\n\n'
        '⛔️【分镜产出绝对红线】⛔️：\n'
        f'1. **强制一一对应**：我传入了 {n} 张图片，你的输出数组中就【必须】精确包含 {n} 个对象！\n'
        '2. **台词局部拆分**：长台词跨越多个画面时拆成逻辑分句，坚决杜绝无语义细碎短语。若是画外音请标注"(画外音)"。\n'
        '3. **严禁凭空捏造台词**：没有识别到说话内容时，dialogue/speaker/emotion 必须填 null。\n'
        '4. **严禁心理活动与文学词汇**：严禁"仿佛"、"似乎"、"意味深长"，只描写摄像机能拍到的物理动作。\n'
        '5. **严禁description包含字幕文字**：字幕内容只进 dialogue，不进任何 description 字段。\n\n'
        '【各字段填写规范】\n'
        '▸ shot_scale：从 [大特写/特写/近景/中近景/中景/全景/远景] 中选一个。\n'
        '▸ camera_angle：从 [俯拍/仰拍/平拍/低角度/斜角/过肩] 中选一个。\n'
        '▸ camera_motion：从 [固定/推进/拉远/摇镜/跟拍/升降] 中选一个。\n'
        '▸ subjects：画面中每个可见人物填写一项，【必须填写人名和性别】。\n'
        '  - name：人物姓名，与人物表一致；无法识别的边缘角色保持全集命名一致，完全无法判断则填"未知"。\n'
        '  - gender：性别，填 男/女/未知。\n'
        '  - clothing：服装颜色+款式（是区分不同人物的辅助标识，必须填写）。\n'
        '  - position：画面中的位置（画面左侧/右侧/中央/前景/背景/左前景/右后景 等）。\n'
        '  - action：具体肢体动作，包含身体部位+动作方向+力度。\n'
        '  - expression：眉毛+眼睛+嘴巴三个部位各一条微表情描述。\n'
        '▸ background：场景类型+主要道具+光线/色调，一句话，不超过30字。\n'
        '▸ description_narrative：结合人物表中的人名，将上述所有信息写成一段流畅的叙事描述，'
        '包含景别、镜头运动、人物姓名+动作+表情、背景，供视频生成模型使用。\n'
        '▸ speaker：dialogue 对应的说话人姓名，与人物表一致；无台词则 null。\n\n'
        '【细节刻画要求（必须执行）】\n'
        '- 微表情：精准描写眉毛（微蹙/舒展）、眼睛（直视/躲闪/视线下移/睁大）、嘴巴（紧闭/微张/嘴角上扬）。\n'
        '- 微动作：描写肢体的质感、倾向和力度，如"双手手指交叉紧紧扣在桌面上"而非"双手放在桌上"。\n\n'
        '【画外音防错规则】\n'
        '- 警惕"视觉欺骗"：如果基于【人物表和潜台词】判定台词属于角色B，但画面里是角色A在倾听，这属于B的画外音！\n'
        '- dialogue 字段必须填真正的说话人(画外音): 完整台词内容。\n\n'
        '- description_narrative 和 dialogue/speaker 中的人名必须与人物表一致。\n'
        '- 边缘角色（不在人物表中）在 description_narrative/dialogue 中的命名需全集一致。\n'
        '- subjects 中的 name 必须与人物表一致；无法识别的边缘角色需全集命名一致。\n'
        '- 相邻分镜严禁出现完全相同的 dialogue！同一句话延续或无台词时必须填 null。\n\n'
        '【输出格式硬性要求（必须严格执行）】\n'
        f'- 最终输出必须是长度为 {n} 的 JSON 数组。\n'
        '- 每个对象必须包含且仅包含以下字段：\n'
        '  index, shot_scale, camera_angle, camera_motion, subjects,\n'
        '  background, description_narrative,\n'
        '  dialogue, speaker, emotion, duration\n'
        '- subjects 是数组，每个人物一项，包含 name/gender/clothing/position/action/expression。\n'
        '- dialogue/speaker/emotion 无内容时填 null。\n'
        '- 不要输出任何解释、说明、前言、markdown 代码块，只输出 JSON 数组。\n'
        '【你必须严格按此骨架（仅展示前2项）输出，共需输出所有图片】\n'
        f'{output_format_str}\n'
    )

    content = []
    for seq, idx, asr_text, duration in per_shot_data:
        content.append({"image": file_urls[seq]})
        label = asr_text if asr_text else "（无台词）"
        parts = [f"分镜{idx:03d}"]
        if has_transcripts:
            parts.append(f"台词：{label}")
            parts.append(f"时长：{duration}秒")
        content.append({"text": f"[{' | '.join(parts)}]"})
    # prompt 单独保存，每次 attempt 动态拼接后加入 content

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"  解析失败，第 {attempt + 1}/{max_retries} 次重试...")
            time.sleep(1)
        current_prompt = prompt
        if attempt > 0 and expected_count is not None:
            current_prompt += f"\n\n⚠️ 再次提醒：输出的JSON数组必须恰好包含 {expected_count} 个对象，最大index为 {expected_count - 1}。"
        current_content = content + [{"text": current_prompt}]
        try:
            response = MultiModalConversation.call(  # type: ignore[assignment]
                api_key=api_key,
                model=MODEL_NAME,
                messages=[{"role": "user", "content": current_content}]
            )
            if response.status_code != 200:  # type: ignore[union-attr]
                print(f"  API 错误: {response.message}")  # type: ignore[union-attr]
                continue
            raw_content = response.output.choices[0].message.content  # type: ignore[union-attr]
            if isinstance(raw_content, str):
                raw_text = raw_content
            elif isinstance(raw_content, list) and raw_content:
                raw_text = raw_content[0]["text"]
            else:
                print("  API 返回内容为空，重试...")
                continue
            result = extract_json(raw_text)
            if isinstance(result, list) and len(result) > 0:
                if expected_count is None or len(result) == expected_count:
                    return result
                print(f"  分镜数量不匹配（期望 {expected_count}，实际 {len(result)}），重试...")
        except Exception as e:
            print(f"  调用失败: {e}")

    print(f"  达到最大重试次数 ({max_retries} 次)，分析失败。")
    return None


# ==========================================
# 4. 查找任务
# ==========================================
def find_tasks(root: str):
    """
    递归查找所有 split/<N>/segments_timeline.json，
    返回 list of (index_str, split_subdir, show_dir)
    """
    tasks = []
    for dirpath, _, filenames in os.walk(root):
        if "segments_timeline.json" not in filenames:
            continue
        # dirpath 形如 saved/<show>/split/<N>
        split_dir = os.path.dirname(dirpath)
        if os.path.basename(split_dir) != "split":
            continue
        show_dir = os.path.dirname(split_dir)
        index_str = os.path.basename(dirpath)  # "1", "2", ...

        output_path = os.path.join(show_dir, "shot", f"{index_str}.json")
        if os.path.exists(output_path):
            print(f"  [跳过] 已存在: {output_path}")
            continue

        tasks.append((index_str, dirpath, show_dir))
    return tasks


# ==========================================
# 5. 处理单个任务
# ==========================================
def process_task(task, api_key):
    index_str, split_subdir, show_dir = task

    timeline_path    = os.path.join(split_subdir, "segments_timeline.json")
    scene_json_path  = os.path.join(show_dir, "plot", f"{index_str}.json")
    transcript_path  = os.path.join(show_dir, "transcript", f"fixed_{index_str}.json")
    output_path      = os.path.join(show_dir, "shot", f"{index_str}.json")

    if not os.path.exists(scene_json_path):
        print(f"  [跳过] 缺少 plot 文件: {scene_json_path}")
        return index_str, False

    segments, durations = load_timeline(timeline_path)
    transcripts = load_transcripts(transcript_path)

    # 按 segments_timeline 顺序收集 keyframe 路径
    image_paths = []
    for seg in segments:
        kf = seg.get("keyframe_filename")
        if not kf:
            continue
        kf_path = os.path.abspath(os.path.join(split_subdir, kf)).replace("\\", "/")
        if os.path.exists(kf_path.replace("/", os.sep)):
            image_paths.append(kf_path)

    if not image_paths:
        print(f"  [跳过] 第 {index_str} 集没有找到 keyframe 图片")
        return index_str, False

    show_name = os.path.basename(show_dir)
    print(f"  [分析] {show_name} 第 {index_str} 集，共 {len(image_paths)} 帧...")

    shots = analyze_all(image_paths, scene_json_path, transcripts, durations, api_key, expected_count=len(segments))

    if shots is None:
        print(f"  [失败] 第 {index_str} 集分析失败，不生成文件。")
        return index_str, False

    # ── 静态生成 description_visual（不含人名，供 depth/pose 检索）──
    for shot in shots:
        shot["description_visual"] = build_description_visual(shot)
        # 补全时长（LLM 可能不填）
        idx = shot.get("index", 0)
        if not shot.get("duration"):
            shot["duration"] = durations.get(idx, 0.0)

    os.makedirs(os.path.join(show_dir, "shot"), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=4)

    print(f"  [完成] 第 {index_str} 集 → {output_path}")
    return index_str, True


# ==========================================
# 6. 主流程
# ==========================================
def main():
    if len(sys.argv) < 2:
        print("用法: python image_shot_analyzer.py <saved 文件夹>")
        sys.exit(1)

    root = sys.argv[1]
    tasks = find_tasks(root)

    if not tasks:
        print("未找到任何待处理任务。")
        return

    print(f"找到 {len(tasks)} 个任务，使用 {MAX_WORKERS} 个并发线程...\n")

    keyed_tasks = [
        (t, API_KEYS[i % len(API_KEYS)])
        for i, t in enumerate(tasks)
    ]

    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_task, t, key): t for t, key in keyed_tasks}
        for future in as_completed(futures):
            _, ok = future.result()
            if ok:
                success += 1
            else:
                fail += 1

    print(f"\n完成：{success} 成功，{fail} 失败。")


if __name__ == "__main__":
    main()
