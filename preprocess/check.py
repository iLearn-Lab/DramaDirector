# 检查 shot / transcript / split 三处分镜数量是否一致。
# 全部一致时：将 shot 中每个分镜的 dialogue 替换为 transcript 的台词，并添加 duration 字段。
# 存在不一致时：在根目录生成 error.json 记录详情，不修改任何 shot 文件。

import json
import os
import sys


def load_timeline(timeline_path):
    """返回 {segment_index: duration_sec}"""
    with open(timeline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        seg["segment_index"]: round(seg["end_sec"] - seg["start_sec"], 1)
        for seg in data.get("segments", [])
    }


def load_transcripts(transcript_path):
    """返回 {int(key): text}"""
    with open(transcript_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def find_tasks(root):
    """
    递归找所有 shot/N.json，匹配对应的 split/N/segments_timeline.json 和 transcript/fixed_N.json。
    返回 list of dict，每项包含 show_name, index_str, shot_path, timeline_path, transcript_path
    """
    tasks = []
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) != "shot":
            continue
        show_dir = os.path.dirname(dirpath)
        show_name = os.path.basename(show_dir)
        for fname in filenames:
            if not fname.endswith(".json"):
                continue
            index_str = fname[:-5]
            tasks.append({
                "show_name":       show_name,
                "index_str":       index_str,
                "shot_path":       os.path.join(dirpath, fname),
                "timeline_path":   os.path.join(show_dir, "split", index_str, "segments_timeline.json"),
                "transcript_path": os.path.join(show_dir, "transcript", f"fixed_{index_str}.json"),
            })
    return tasks


def find_missing_shots(root):
    """
    扫描所有 split/<N>/segments_timeline.json，返回缺少对应 shot/<N>.json 的项。
    返回 list of dict，格式与 errors 条目一致。
    """
    missing = []
    for dirpath, _, filenames in os.walk(root):
        if "segments_timeline.json" not in filenames:
            continue
        split_dir = os.path.dirname(dirpath)
        if os.path.basename(split_dir) != "split":
            continue
        show_dir = os.path.dirname(split_dir)
        show_name = os.path.basename(show_dir)
        index_str = os.path.basename(dirpath)
        shot_path = os.path.join(show_dir, "shot", f"{index_str}.json")
        if not os.path.exists(shot_path):
            missing.append({
                "show":    show_name,
                "index":   index_str,
                "error":   "shot文件缺失（分析失败或未运行）",
                "missing": [shot_path],
            })
    return missing


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "saved"
    error_output = os.path.join(os.path.dirname(os.path.abspath(root)), "error.json") \
        if root != "saved" else "error.json"

    errors = []
    ok_tasks = []

    # 先检查有 split 但没有 shot 的遗漏集数
    for item in find_missing_shots(root):
        errors.append(item)
        print(f"  [遗漏] {item['show']} 第 {item['index']} 集：shot 文件不存在")

    tasks = find_tasks(root)
    if not tasks and not errors:
        print("未找到任何 shot/*.json 文件。")
        return

    for t in tasks:
        label = f"{t['show_name']} 第 {t['index_str']} 集"

        # 检查文件是否存在
        missing = [p for p in (t["shot_path"], t["timeline_path"], t["transcript_path"])
                   if not os.path.exists(p)]
        if missing:
            errors.append({
                "show":  t["show_name"],
                "index": t["index_str"],
                "error": "文件缺失",
                "missing": missing,
            })
            print(f"  [缺失] {label}: {missing}")
            continue

        # 读取三处数量
        with open(t["shot_path"], "r", encoding="utf-8") as f:
            shot_data = json.load(f)
        timeline = load_timeline(t["timeline_path"])
        transcripts = load_transcripts(t["transcript_path"])

        n_shot       = len(shot_data)
        n_timeline   = len(timeline)
        n_transcript = len(transcripts)

        if n_shot == n_timeline == n_transcript:
            ok_tasks.append((t, shot_data, timeline, transcripts))
            print(f"  [OK]  {label}：{n_shot} 个分镜")
        else:
            deleted = []
            # transcript 数量错误：删除 fixed_N.json 和 shot/N.json
            if n_transcript != n_timeline:
                if os.path.exists(t["transcript_path"]):
                    os.remove(t["transcript_path"])
                    deleted.append(t["transcript_path"])
                if os.path.exists(t["shot_path"]):
                    os.remove(t["shot_path"])
                    deleted.append(t["shot_path"])
            # shot 数量错误（transcript 正确）：只删除 shot/N.json
            elif n_shot != n_timeline:
                if os.path.exists(t["shot_path"]):
                    os.remove(t["shot_path"])
                    deleted.append(t["shot_path"])
            errors.append({
                "show":               t["show_name"],
                "index":              t["index_str"],
                "error":              "分镜数量不一致",
                "shot_count":         n_shot,
                "timeline_count":     n_timeline,
                "transcript_count":   n_transcript,
                "deleted":            deleted,
            })
            print(f"  [不一致] {label}：shot={n_shot}  timeline={n_timeline}  transcript={n_transcript}")
            for d in deleted:
                print(f"    [删除] {d}")

    # 如果有任何错误，写 error.json
    if errors:
        with open(error_output, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\n发现 {len(errors)} 处不一致，详见 {error_output}。")

    if not ok_tasks:
        return

    # 更新一致的 shot 文件
    print(f"\n开始更新 {len(ok_tasks)} 个一致的 shot 文件...")
    for t, shot_data, timeline, transcripts in ok_tasks:
        for seq, item in enumerate(shot_data):
            if "index" not in item:
                item["index"] = seq
            idx = int(item["index"])
            item["dialogue"] = transcripts.get(idx, "")
            item["duration"] = round(timeline.get(idx, 0.0), 1)

        with open(t["shot_path"], "w", encoding="utf-8") as f:
            json.dump(shot_data, f, ensure_ascii=False, indent=4)
        print(f"  [更新] {t['show_name']} 第 {t['index_str']} 集 → {t['shot_path']}")

    print(f"\n完成，共更新 {len(ok_tasks)} 个文件。")


if __name__ == "__main__":
    main()
