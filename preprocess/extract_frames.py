import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from shot_split import ShotSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DEFAULT_SAVED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".ts", ".m4v"}
ROOT_DIR = Path(__file__).resolve().parent
SAVED_DIR = DEFAULT_SAVED_DIR
MAX_WORKERS = 8


def process_video(splitter: ShotSplitter, video_file: Path, subfolder_name: str, index: int) -> None:
    output_dir = SAVED_DIR / subfolder_name / "split" / str(index)
    timeline_file = output_dir / "segments_timeline.json"
    if output_dir.exists() and timeline_file.exists():
        logger.info(f"[跳过] 已存在: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[处理] {video_file.name} -> {output_dir}")
    try:
        shots = splitter.split_video(str(video_file), str(output_dir))
        logger.info(f"[完成] {video_file.name}: {len(shots)} 个镜头")
    except Exception as e:
        logger.error(f"[失败] {video_file.name}: {e}")


def extract_all_videos(input_folder: str) -> None:
    """
    扫描 input_folder 的直接子文件夹，并发分割其中的视频。
    输出路径: saved/<子文件夹名>/split/<视频索引>/
    """
    input_path = Path(input_folder).resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"找不到文件夹: {input_folder}")

    # 收集所有任务：(视频路径, 子文件夹名, 索引)
    tasks = []
    for subdir in sorted(input_path.iterdir()):
        if not subdir.is_dir():
            continue
        video_files = sorted(
            (f for f in subdir.iterdir()
             if f.suffix.lower() in VIDEO_EXTENSIONS),
            key=lambda f: int(f.stem) if f.stem.isdigit() else f.name,
        )
        for idx, video_file in enumerate(video_files, start=1):
            # 若文件名本身是纯数字（如 10.mp4），直接用文件名作为索引
            idx = int(video_file.stem) if video_file.stem.isdigit() else idx
            tasks.append((video_file, subdir.name, idx))

    if not tasks:
        logger.warning(f"在 {input_folder} 的子文件夹中未找到视频文件")
        return

    logger.info(f"找到 {len(tasks)} 个视频，使用 {MAX_WORKERS} 线程并发处理...")

    splitter = ShotSplitter(threshold=30.0, save_keyframe=True, min_scene_len=15)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_video, splitter, vf, sn, idx): vf
            for vf, sn, idx in tasks
        }
        for future in as_completed(futures):
            vf = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"[错误] {vf.name}: {e}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "videos"
    extract_all_videos(folder)
