# 输入：包含 split 子目录的文件夹（支持递归）。输出：台词 JSON。
# 用法：python transcribe.py <root_dir>
#   递归查找 root_dir 下所有 split/ 目录，
#   读取 split/<N>/segments_timeline.json，按分镜对原始视频切割音频转录，
#   保存到同级 transcript/<N>/transcripts.json

import sys
import os
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─── 将 Fun-ASR 目录加入模块搜索路径 ───────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "Fun-ASR"))
from model import FunASRNano  # noqa: E402  位于 Fun-ASR/model.py

# ==========================================
# 配置
# ==========================================
ASR_MODEL_DIR   = os.path.join(_ROOT, "Fun-ASR", "FunAudioLLM", "Fun-ASR-Nano-2512")
ASR_DEVICE      = "cuda:0"
EXTRACT_WORKERS = 12   # ffmpeg 并发提取音频线程数


# ==========================================
# 核心函数
# ==========================================
def resolve_source_video(raw_path: str) -> str:
    """
    将 segments_timeline.json 中记录的 source_video 路径解析为当前系统可用路径。
    支持跨平台：若 Windows 绝对路径在 Linux 上不存在，
    则提取路径中 'videos' 目录及之后的部分，拼接到脚本根目录重新查找。
    """
    if os.path.exists(raw_path):
        return raw_path
    # 统一分隔符后分割，找到 'videos' 片段
    parts = raw_path.replace("\\", "/").split("/")
    try:
        idx = next(i for i, p in enumerate(parts) if p.lower() == "videos")
    except StopIteration:
        return raw_path  # 无法解析，原样返回（后续会报错）
    rel = os.path.join(*parts[idx:])
    candidate = os.path.join(_ROOT, rel)
    if os.path.exists(candidate):
        return candidate
    return raw_path  # 都找不到，原样返回（后续报错提示原路径）


def copy_to_ascii(src: str, tmpdir: str, name: str) -> str:
    """把含中文路径的文件复制到临时目录（ASCII路径），返回新路径。"""
    ext = os.path.splitext(src)[1]
    dst = os.path.join(tmpdir, f"{name}{ext}")
    shutil.copy2(src, dst)
    return dst


def extract_audio_segment(source_video: str, start_sec: float, duration: float,
                           audio_path: str) -> None:
    """从视频按时间段截取音频，输出 16kHz 单声道 wav。"""
    result = subprocess.run(
        ["ffmpeg", "-y",
         "-ss", str(start_sec),
         "-i", source_video,
         "-t", str(duration),
         "-ar", "16000", "-ac", "1", "-vn",
         audio_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败:\n{result.stderr}")


def transcribe_sub(split_sub: str, asr_model, kwargs: dict, output_path: str) -> None:
    """
    处理 split/<N>/ 一个子目录：
      读取 segments_timeline.json，找到原始视频，按分镜切割音频转录。
      结果保存到 output_path（JSON）。
    """
    if os.path.exists(output_path):
        tqdm.write(f"  [跳过] 已存在: {output_path}")
        return

    timeline_json = os.path.join(split_sub, "segments_timeline.json")
    if not os.path.exists(timeline_json):
        tqdm.write(f"  [跳过] 无 segments_timeline.json: {split_sub}")
        return

    with open(timeline_json, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    source_video = resolve_source_video(timeline["source_video"])
    segments     = timeline["segments"]

    if not os.path.exists(source_video):
        tqdm.write(f"  [错误] 原始视频不存在: {source_video}")
        return

    results: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Windows 下 ffmpeg 不支持中文路径，复制原始视频到临时目录
        tmp_video = copy_to_ascii(source_video, tmpdir, "source")

        # 1. 并行提取所有分镜音频
        def extract_seg(seg):
            idx        = seg["segment_index"]
            start_sec  = seg["start_sec"]
            duration   = seg["end_sec"] - seg["start_sec"]
            audio_path = os.path.join(tmpdir, f"seg_{idx:04d}.wav")
            extract_audio_segment(tmp_video, start_sec, duration, audio_path)
            return idx, audio_path

        seg_label = os.path.basename(split_sub)
        ordered_pairs: list[tuple[int, str | None]] = [None] * len(segments)  # type: ignore[list-item]
        failed_idxs: set[int] = set()

        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
            futures = {pool.submit(extract_seg, seg): i for i, seg in enumerate(segments)}
            with tqdm(total=len(segments), desc=f"    {seg_label} 提取", unit="seg", leave=False) as pbar:
                for future in as_completed(futures):
                    pos = futures[future]
                    try:
                        idx, audio_path = future.result()
                        ordered_pairs[pos] = (idx, audio_path)
                    except Exception as e:
                        idx = segments[pos]["segment_index"]
                        tqdm.write(f"    [{idx:04d}] 提取失败: {e}")
                        failed_idxs.add(idx)
                        ordered_pairs[pos] = (idx, None)
                    pbar.update(1)

        # 2. 逐个 ASR 推理
        valid_pairs  = [(idx, p) for idx, p in ordered_pairs if p is not None]
        invalid_idxs = {idx for idx, p in ordered_pairs if p is None}

        if valid_pairs:
            with tqdm(total=len(valid_pairs), desc=f"    {seg_label} 推理", unit="seg", leave=False) as pbar:
                for idx, audio_path in valid_pairs:
                    try:
                        res = asr_model.inference(data_in=[audio_path], **kwargs)
                        text = res[0][0].get("text", "").strip() if res and res[0] else ""
                    except Exception as e:
                        tqdm.write(f"    [{idx:04d}] 推理失败: {e}")
                        text = ""
                    results[str(idx)] = text
                    pbar.update(1)

        for idx in invalid_idxs:
            results[str(idx)] = ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    nonempty = sum(1 for v in results.values() if v)
    tqdm.write(f"  [完成] {os.path.basename(split_sub)} — {nonempty}/{len(results)} 个有台词 → {output_path}")


def find_split_dirs(root: str) -> list[str]:
    """递归查找 root 下所有名为 'split' 的目录（不进入 split 内部继续找）。"""
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == "split":
            found.append(dirpath)
            dirnames.clear()
        else:
            dirnames[:] = sorted(dirnames)
    return sorted(found)


def process_split_dir(split_dir: str, asr_model, kwargs: dict) -> None:
    """
    处理一个 split 目录下的所有子目录：
      split/<N>/  →  transcript/<N>/transcripts.json
    """
    show_dir        = os.path.dirname(split_dir)
    transcript_root = os.path.join(show_dir, "transcript")

    subs = sorted(
        e for e in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, e))
    )
    if not subs:
        print(f"  [跳过] split 目录下无子目录: {split_dir}")
        return

    for sub in subs:
        sub_path    = os.path.join(split_dir, sub)
        output_path = os.path.join(transcript_root, f"{sub}.json")
        transcribe_sub(sub_path, asr_model, kwargs, output_path)


# ==========================================
# 主流程
# ==========================================
def main():
    if len(sys.argv) < 2:
        print("用法: python transcribe.py <root_dir>")
        print("  root_dir 可以是单个剧集目录，也可以是包含多个剧集的父目录。")
        sys.exit(1)

    root_dir = sys.argv[1].rstrip("/\\")
    if not os.path.isdir(root_dir):
        print(f"[错误] 路径不存在或不是目录: {root_dir}")
        sys.exit(1)

    split_dirs = find_split_dirs(root_dir)
    if not split_dirs:
        print(f"[错误] 在 {root_dir} 下未找到任何 split 目录")
        sys.exit(1)

    print(f"找到 {len(split_dirs)} 个 split 目录，加载 FunASR 模型...")
    asr_model, kwargs = FunASRNano.from_pretrained(
        model=ASR_MODEL_DIR, device=ASR_DEVICE
    )
    asr_model.eval()

    for split_dir in split_dirs:
        print(f"\n处理: {split_dir}")
        process_split_dir(split_dir, asr_model, kwargs)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
