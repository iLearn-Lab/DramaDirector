import os
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIMELINE_MANIFEST_FILE = "segments_timeline.json"

def _compute_frame_diffs(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Compute HSV histogram differences between adjacent frames.

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        Array of frame differences, length = total_frames - 1
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        return np.array([])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    if not ret:
        return np.array([])

    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    diffs = []

    for _ in range(1, total_frames):
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)

        # Calculate H and S channel histogram differences
        h_diff = cv2.absdiff(prev_hsv[:, :, 0], curr_hsv[:, :, 0]).mean()
        s_diff = cv2.absdiff(prev_hsv[:, :, 1], curr_hsv[:, :, 1]).mean()

        diffs.append(h_diff + s_diff)
        prev_hsv = curr_hsv

    return np.array(diffs)


def _detect_scenes(
    diffs: np.ndarray, threshold: float = 30.0, min_scene_len: int = 15
) -> List[int]:
    """
    Detect scene cut points using local maxima detection.

    Strategy:
    1. Find all local maxima (peaks)
    2. Filter peaks below threshold
    3. Filter cut points that are too close (keep higher scoring ones)

    Args:
        diffs: Frame difference array
        threshold: Detection threshold
        min_scene_len: Minimum distance between cut points

    Returns:
        List of cut point frame indices (0-indexed, referring to inter-frame positions)
    """
    if len(diffs) == 0:
        return []

    # Find local maxima (peaks)
    peaks = []
    for i in range(1, len(diffs) - 1):
        if diffs[i] > diffs[i - 1] and diffs[i] > diffs[i + 1]:
            peaks.append((i, diffs[i]))

    # Sort by score
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Select peaks ensuring minimum spacing
    cut_points = []
    for idx, score in peaks:
        if score < threshold:
            continue

        # Check distance from already selected cut points
        too_close = False
        for cut in cut_points:
            if abs(idx - cut) < min_scene_len:
                too_close = True
                break

        if not too_close:
            cut_points.append(idx)

    return sorted(cut_points)

def _analyze_video_metrics(cap: cv2.VideoCapture) -> Tuple[np.ndarray, np.ndarray]:
    """
    同时计算视频的 HSV 帧差（用于场景检测）和 亮度值（用于闪光检测）。
    
    Returns:
        (diffs, brightness): 两个 numpy 数组
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        return np.array([]), np.array([])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    if not ret:
        return np.array([]), np.array([])

    # 初始化上一帧的 HSV
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    
    # 记录第一帧亮度
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    brightness_values = [np.mean(prev_gray)]
    
    diffs = []

    for _ in range(1, total_frames):
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 1. 计算亮度 (Brightness)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(curr_gray)
        brightness_values.append(avg_brightness)

        # 2. 计算 HSV 差异 (Scene Diff)
        curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        
        # H 和 S 通道的差异
        h_diff = cv2.absdiff(prev_hsv[:, :, 0], curr_hsv[:, :, 0]).mean()
        s_diff = cv2.absdiff(prev_hsv[:, :, 1], curr_hsv[:, :, 1]).mean()
        
        diffs.append(h_diff + s_diff)
        prev_hsv = curr_hsv

    return np.array(diffs), np.array(brightness_values)


def _detect_scene_cuts(
    diffs: np.ndarray, threshold: float = 30.0, min_scene_len: int = 15
) -> List[int]:
    """
    基于 HSV 差异检测常规场景切换。
    """
    if len(diffs) == 0:
        return []

    # Find local maxima (peaks)
    peaks = []
    for i in range(1, len(diffs) - 1):
        if diffs[i] > diffs[i - 1] and diffs[i] > diffs[i + 1]:
            peaks.append((i, diffs[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)

    cut_points = []
    for idx, score in peaks:
        if score < threshold:
            continue
        
        # 距离过滤
        if all(abs(idx - cut) >= min_scene_len for cut in cut_points):
            cut_points.append(idx)

    return sorted(cut_points)


def _detect_flash_cuts(
    brightness: np.ndarray, flash_threshold: float = 230.0, min_scene_len: int = 15
) -> Tuple[List[int], Dict[int, int]]:
    """
    基于高亮帧检测闪光转场。
    策略：找到连续高亮区域的峰值作为切点，并后移若干帧以保证下一段首帧有效。

    Returns:
        (final_cuts, flash_origins):
            final_cuts - 后移后的切点帧索引列表
            flash_origins - 映射 {后移后的切点: 原始峰值位置}，
                            用于将前段视频中后移出来的尾帧设为白色
    """
    if len(brightness) == 0:
        return [], {}

    # 找到所有超过阈值的帧索引
    high_bright_indices = np.where(brightness > flash_threshold)[0]

    if len(high_bright_indices) == 0:
        return [], {}

    flash_cuts = []
    flash_origins: Dict[int, int] = {}  # shifted_cut -> original_peak
    # 将连续的帧归为一组（一个闪光过程可能持续几帧）
    if len(high_bright_indices) > 0:
        # 简单的聚类：如果索引不连续，视为新的闪光
        current_group = [high_bright_indices[0]]

        for i in range(1, len(high_bright_indices)):
            if high_bright_indices[i] == high_bright_indices[i-1] + 1:
                current_group.append(high_bright_indices[i])
            else:
                # 处理上一组，找到最亮的那个点后，选取后面的正常画面作为切割点
                original_peak = max(current_group, key=lambda idx: brightness[idx])
                best_idx = original_peak + 20
                flash_cuts.append(best_idx)
                flash_origins[best_idx] = original_peak
                current_group = [high_bright_indices[i]]

        # 处理最后一组（不后移，接近视频末尾）
        if current_group:
            best_idx = max(current_group, key=lambda idx: brightness[idx])
            flash_cuts.append(best_idx)

    # 过滤距离太近的闪光（虽然不太可能连续闪，但为了保险）
    final_cuts = []
    for cut in flash_cuts:
        if not final_cuts or abs(cut - final_cuts[-1]) >= min_scene_len:
            final_cuts.append(cut)

    # 只保留最终通过过滤的切点对应的 origin 信息
    final_origins = {cut: flash_origins[cut] for cut in final_cuts if cut in flash_origins}

    return final_cuts, final_origins
class ShotSplitter:
    """
    A class to split videos into shots based on scene detection using OpenCV.
    Supports processing single videos or all videos in a directory.
    Uses local maxima detection on HSV histogram differences (based on PySceneDetect algorithm).
    """

    def __init__(
        self,
        threshold: float = 30.0,
        save_keyframe: bool = True,
        min_scene_len: int = 15,
        flash_threshold: float = 230.0,
    ):
        """
        Initialize the ShotSplitter.

        Args:
            threshold: Threshold for scene detection. Higher values detect fewer scenes.
            save_keyframe: Whether to save the first frame (keyframe) of each split video as a PNG image.
            min_scene_len: Minimum number of frames between scene cuts.
            flash_threshold: Threshold for flash detection based on brightness.
        """
        self.threshold = threshold
        self.save_keyframe = save_keyframe
        self.min_scene_len = min_scene_len
        self.flash_threshold = flash_threshold
        logger.info(
            f"ShotSplitter initialized with threshold: {threshold}, save_keyframe: {save_keyframe}, min_scene_len: {min_scene_len}, flash_threshold: {flash_threshold}"
        )

    def split_video(
        self,
        video_path: str,
        output_dir: str,
        output_template: str = "video_$SCENE_NUMBER.mp4",
        keyframe_template: str = "keyframe_$SCENE_NUMBER.png",
    ) -> List[str]:
        """
        Split a single video into shots using OpenCV-based scene detection.

        Args:
            video_path: Path to the input video file
            output_dir: Directory to save the split video files
            output_template: Template for output video filenames (default: "video_$SCENE_NUMBER.mp4")
            keyframe_template: Template for keyframe filenames (default: "keyframe_$SCENE_NUMBER.png")

        Returns:
            List of output video file paths

        Raises:
            FileNotFoundError: If video file doesn't exist
            Exception: If splitting fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames"
        )

        # Compute frame difference curve
        # logger.info("Computing frame differences...")
        # diffs = _compute_frame_diffs(cap)

        # 1. 计算所有指标
        logger.info("Computing frame metrics (HSV & Brightness)...")
        diffs, brightness = _analyze_video_metrics(cap)

        if len(diffs) == 0:
            logger.error("Cannot compute frame differences")
            cap.release()
            return []

        # 2. 检测常规场景切换
        mean_diff = diffs.mean()
        auto_threshold = max(self.threshold, mean_diff * 2.5)
        hsv_cuts = _detect_scene_cuts(diffs, auto_threshold, self.min_scene_len)
        logger.info(f"Detected {len(hsv_cuts)} scene cuts (HSV)")
        # Adaptive threshold (based on v6's approach)
        # mean_diff = diffs.mean()
        # auto_threshold = max(self.threshold, mean_diff * 2.5)
        # logger.info(
        #     f"Adaptive threshold: {auto_threshold:.2f} (base={self.threshold}, mean={mean_diff:.2f})"
        # )

        # 3. 检测闪光转场
        flash_cuts, flash_origins = _detect_flash_cuts(brightness, self.flash_threshold, self.min_scene_len)
        if flash_cuts:
            logger.info(f"Detected {len(flash_cuts)} flash cuts at frames: {flash_cuts}")

        # hsv_cuts 是 diffs 的索引，转换为帧索引需要 +1
        hsv_frame_indices = [idx + 1 for idx in hsv_cuts]

        combined_cuts = sorted(list(set(hsv_frame_indices + flash_cuts)))

        # 5. 最终过滤（确保合并后的点也满足最小距离）
        final_cuts = []
        if combined_cuts:
            final_cuts.append(combined_cuts[0])
            for cut in combined_cuts[1:]:
                if cut - final_cuts[-1] >= self.min_scene_len:
                    final_cuts.append(cut)

        logger.info(f"Total cut points after merge: {len(final_cuts)}")

        # 构建完整的切分列表
        all_cuts = [0] + final_cuts + [total_frames]
        # Detect scenes
        # logger.info("Detecting scenes...")
        # cut_points = _detect_scenes(diffs, auto_threshold, self.min_scene_len)

        # if len(cut_points) == 0:
        #     logger.warning("No scene cuts detected")
        #     cap.release()
        #     return []

        # logger.info(f"Detected {len(cut_points)} cut points")

        # # Convert inter-frame indices to actual frame indices
        # # diffs[i] represents difference between frame i and i+1
        # # So cut at index i should be at frame i+1
        # cut_points = [idx + 1 for idx in cut_points]

        # # Build complete cut point list: [0, cut1, cut2, ..., total_frames]
        # all_cuts = [0] + cut_points + [total_frames]

        # Split video
        output_files = []
        timeline_segments: List[Dict[str, object]] = []
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 预创建白色帧，用于替换闪光后移区域
        white_frame = np.full((height, width, 3), 255, dtype=np.uint8)

        for i in range(len(all_cuts) - 1):
            start_frame = all_cuts[i]
            end_frame = all_cuts[i + 1]
            num_frames = end_frame - start_frame

            if num_frames < self.min_scene_len:
                logger.debug(f"Skipping short shot {i}: {num_frames} frames")
                continue

            # 判断本段末尾是否为闪光后移切点，若是则获取原始峰值位置
            white_from_frame = flash_origins.get(end_frame)
            if white_from_frame is not None:
                white_count = end_frame - white_from_frame
                logger.info(
                    f"Shot {i}: flash shift detected, whiting out last {white_count} frames "
                    f"(frame {white_from_frame} ~ {end_frame - 1})"
                )

            # Use template for output filename (0-indexed, 3 digits)
            video_filename = output_template.replace("$SCENE_NUMBER", f"{i:03d}")
            output_path = os.path.join(output_dir, video_filename)
            keyframe_filename = keyframe_template.replace("$SCENE_NUMBER", f"{i:03d}")

            duration = num_frames / fps
            logger.info(
                f"Writing shot {i}: {video_filename} ({duration:.2f}s, {num_frames} frames)"
            )

            # Read first frame for keyframe saving
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, first_frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read first frame at position {start_frame}")
                continue

            # Save keyframe if enabled
            if self.save_keyframe:
                keyframe_path = os.path.join(output_dir, keyframe_filename)
                cv2.imencode(".png", first_frame)[1].tofile(keyframe_path)
                logger.debug(f"Saved keyframe to: {keyframe_path}")

            # Write video segment
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 写入首帧（闪光白化区域在段尾，首帧不受影响）
            current_frame_idx = start_frame
            if white_from_frame is not None and current_frame_idx >= white_from_frame:
                writer.write(white_frame)
            else:
                writer.write(first_frame)

            frames_written = 1
            for _ in range(num_frames - 1):
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_idx += 1
                # 闪光后移区域：将帧替换为白色
                if white_from_frame is not None and current_frame_idx >= white_from_frame:
                    writer.write(white_frame)
                else:
                    writer.write(frame)
                frames_written += 1

            writer.release()
            output_files.append(output_path)
            timeline_segments.append(
                {
                    "segment_index": i,
                    "video_filename": video_filename,
                    "keyframe_filename": keyframe_filename,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "num_frames": int(num_frames),
                    "start_sec": round(float(start_frame) / float(fps), 6),
                    "end_sec": round(float(end_frame) / float(fps), 6),
                    "fps": float(fps),
                }
            )
            logger.info(f"  Wrote {frames_written} frames")

        cap.release()
        manifest_path = Path(output_dir) / TIMELINE_MANIFEST_FILE
        try:
            payload = {
                "version": 1,
                "source_video": str(video_path),
                "fps": float(fps),
                "total_frames": int(total_frames),
                "segments": timeline_segments,
            }
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to write timeline manifest %s: %s", manifest_path, exc)
        logger.info(f"Completed! Generated {len(output_files)} shots")
        return output_files

    def split_videos_in_directory(
        self,
        input_dir: str,
        output_dir: str,
        video_extensions: Optional[List[str]] = None,
    ) -> dict:
        """
        Split all videos in a directory into shots.

        Args:
            input_dir: Directory containing input video files
            output_dir: Base directory to save split videos (subdirectories will be created for each video)
            video_extensions: List of video file extensions to process (default: ['.mp4', '.avi', '.mov', '.mkv'])

        Returns:
            Dictionary mapping original video paths to lists of split video paths

        Raises:
            NotADirectoryError: If input_dir is not a directory
        """
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        if video_extensions is None:
            video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Scanning directory: {input_dir}")

        # Find all video files
        video_files = []
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(input_dir, file))

        logger.info(f"Found {len(video_files)} video files to process")

        # Process each video
        results = {}
        for video_path in video_files:
            video_name = Path(video_path).stem
            video_output_dir = os.path.join(output_dir, video_name)

            try:
                output_files = self.split_video(video_path, video_output_dir)
                results[video_path] = output_files
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results[video_path] = []

        logger.info(f"Completed processing {len(video_files)} videos")
        return results

    def __call__(self, input_path: str, output_dir: str) -> dict:
        """
        Callable interface for splitting videos.
        Automatically detects if input is a file or directory.

        Args:
            input_path: Path to a video file or directory containing videos
            output_dir: Directory to save the split videos

        Returns:
            Dictionary mapping original video paths to lists of split video paths
        """
        if os.path.isfile(input_path):
            # Single video file
            output_files = self.split_video(input_path, output_dir)
            return {input_path: output_files}
        elif os.path.isdir(input_path):
            # Directory of videos
            return self.split_videos_in_directory(input_path, output_dir)
        else:
            raise ValueError(
                f"Input path is neither a file nor a directory: {input_path}"
            )


def load_shots_from_directory(
    shots_directory: str,
    video_extensions: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Load pre-split video shots from a directory structure.

    Expected directory structure:
        shots_directory/
        ├── video1_name/
        │   ├── video_000.mp4
        │   ├── video_001.mp4
        │   └── video_002.mp4
        └── video2_name/
            ├── video_000.mp4
            └── video_001.mp4

    Args:
        shots_directory: Directory containing subdirectories with video shots
        video_extensions: List of video file extensions to process (default: ['.mp4', '.avi', '.mov', '.mkv'])
        verbose: Whether to print progress information (default: True)

    Returns:
        Dictionary mapping subdirectory names to lists of shot file paths (sorted)
        Format: {"video1_name": ["video1_name/video_000.mp4", ...], ...}

    Raises:
        NotADirectoryError: If shots_directory doesn't exist or is not a directory

    Example:
        >>> split_results = load_shots_from_directory("original_shots/")
        >>> analyze_shots(split_results)
    """
    if not os.path.isdir(shots_directory):
        raise NotADirectoryError(f"Shots directory not found: {shots_directory}")

    if video_extensions is None:
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

    if verbose:
        print("\n" + "=" * 60)
        print("Loading Shots from Directory")
        print("=" * 60 + "\n")
        print(f"Directory: {shots_directory}\n")

    split_results = {}

    # Scan subdirectories
    for subdir in os.listdir(shots_directory):
        subdir_path = os.path.join(shots_directory, subdir)

        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue

        # Find all video files in subdirectory
        video_files = []
        for file in os.listdir(subdir_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(subdir_path, file))

        # Sort video files by name to preserve order
        video_files.sort()

        if video_files:
            # Use subdirectory name as the "original video" identifier
            split_results[subdir] = video_files

            if verbose:
                print(f"  Found {len(video_files)} shot(s) in: {subdir}")

    total_shots = sum(len(shots) for shots in split_results.values())

    if verbose:
        print("\n" + "=" * 60)
        print(
            f"Loaded {len(split_results)} video set(s) with {total_shots} total shot(s)"
        )
        print("=" * 60 + "\n")

    return split_results


def load_original_shots(
    input_video_path: str,
    output_shots_dir: str = ".working_dir/original_shots",
    threshold: float = 30.0,
    save_keyframe: bool = True,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Load original videos and split them into shots based on scene detection using OpenCV.

    Args:
        input_video_path: Path to a single video file or directory containing videos
        output_shots_dir: Directory to save split shots (default: ".working_dir/original_shots")
        threshold: Scene detection threshold (default: 30.0)
        save_keyframe: Whether to save the first frame (keyframe) of each split video as a PNG image (default: True)
        verbose: Whether to print progress information (default: True)

    Returns:
        Dictionary mapping original video paths to lists of split shot file paths

    Raises:
        ValueError: If input_video_path is empty or invalid
        Exception: If splitting fails

    Example:
        >>> results = load_original_shots(
        ...     input_video_path="videos/my_video.mp4",
        ...     output_shots_dir=".working_dir/shots",
        ...     save_keyframe=True
        ... )
    """
    if not input_video_path:
        raise ValueError("input_video_path cannot be empty")

    if verbose:
        print("\n" + "=" * 60)
        print("Step 1: Splitting Videos into Shots")
        print("=" * 60 + "\n")
        print(f"Input: {input_video_path}")
        print(f"Output: {output_shots_dir}")
        print(f"Threshold: {threshold}")
        print(f"Save keyframe: {save_keyframe}\n")

    splitter = ShotSplitter(threshold=threshold, save_keyframe=save_keyframe)
    split_results = splitter(input_video_path, output_shots_dir)

    total_shots = sum(len(shots) for shots in split_results.values())

    if verbose:
        print("\n" + "=" * 60)
        print(f"Completed! Split into {total_shots} shot(s)")
        if save_keyframe:
            print(f"Keyframes saved for each shot")
        print("=" * 60 + "\n")

    return split_results
