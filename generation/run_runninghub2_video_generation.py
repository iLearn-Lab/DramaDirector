from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_runninghub_api_key


DEFAULT_UPLOAD_ENDPOINT = "https://www.runninghub.cn/openapi/v2/media/upload/binary"
DEFAULT_GENERATE_ENDPOINT = "https://www.runninghub.cn/openapi/v2/vidu/image-to-video-q3-turbo"
DEFAULT_QUERY_ENDPOINT = "https://www.runninghub.cn/openapi/v2/query"
DEFAULT_PROMPT_SOURCE_TEMPLATE = (
    "collected_prepared_shot_count_{count}/video_inputs/infer_results_rl_grpo_qwen3_8b_all.repaired"
)
SUCCESS_STATES = {"SUCCESS"}
FAILED_STATES = {"FAILED"}
RUNNING_STATES = {"QUEUED", "RUNNING"}


@dataclass(frozen=True)
class ShotTask:
    shot_count: str
    sample_name: str
    shot_name: str
    first_frame_path: Path
    generation_result_path: Path

    @property
    def shot_id(self) -> str:
        return f"{self.shot_count}/{self.sample_name}/{self.shot_name}"


@dataclass(frozen=True)
class PromptEntry:
    prompt: str
    duration_raw: Any
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos from runninghub assets by scanning sample shot folders."
    )
    parser.add_argument(
        "--assets-root",
        type=str,
        default="runninghub_outputs 2",
        help="Root directory containing shot_count_x/sample_xxx/shots/shot_xxx folders.",
    )
    parser.add_argument(
        "--shot-counts",
        type=str,
        default="",
        help="Comma-separated shot_count filters, e.g. 5,6. Empty means all shot_count_* folders.",
    )
    parser.add_argument(
        "--sample-names",
        type=str,
        default="",
        help="Comma-separated sample folder names, e.g. sample_255_s255,sample_497_s497.",
    )
    parser.add_argument(
        "--sample-pattern",
        type=str,
        default="sample_*",
        help="Glob pattern for sample folders when --sample-names is not used.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Limit number of samples to process after filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--sample-select",
        type=str,
        default="first",
        choices=["first", "random"],
        help="How to pick samples when --sample-limit is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of shots to process after filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max parallel workers.",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="5",
        help="Fallback duration enum string used only when source duration is missing.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=["540p", "720p", "1080p"],
        help="Output resolution.",
    )
    parser.add_argument(
        "--audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable audio generation. Default is disabled.",
    )
    parser.add_argument(
        "--prompt-source-template",
        type=str,
        default=DEFAULT_PROMPT_SOURCE_TEMPLATE,
        help=(
            "Template for prompt/duration source directory. Use {count} placeholder, "
            "e.g. collected_prepared_shot_count_{count}/video_inputs/infer_results_rl_grpo_qwen3_8b_all.repaired"
        ),
    )
    parser.add_argument(
        "--generate-endpoint",
        type=str,
        default=DEFAULT_GENERATE_ENDPOINT,
        help="RunningHub image-to-video endpoint.",
    )
    parser.add_argument(
        "--upload-endpoint",
        type=str,
        default=DEFAULT_UPLOAD_ENDPOINT,
        help="RunningHub media upload endpoint.",
    )
    parser.add_argument(
        "--query-endpoint",
        type=str,
        default=DEFAULT_QUERY_ENDPOINT,
        help="RunningHub task query endpoint.",
    )
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-task timeout for polling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runninghub_video_generation_results",
        help="Directory for task records and optional downloads.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download generated file when task succeeds and URL is available.",
    )
    parser.add_argument(
        "--skip-existing-success",
        action="store_true",
        help="Skip shots that already have SUCCESS record in output dir.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries for upload/submit/download and transient polling query errors.",
    )
    parser.add_argument(
        "--retry-initial-delay",
        type=float,
        default=2.0,
        help="Initial retry delay in seconds.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Retry backoff multiplier.",
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=20.0,
        help="Max retry delay in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and extract prompts; do not call API.",
    )
    return parser.parse_args()


def parse_csv(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def list_shot_count_dirs(root: Path, shot_count_filters: list[str]) -> list[Path]:
    if shot_count_filters:
        wanted = {f"shot_count_{x}" if not x.startswith("shot_count_") else x for x in shot_count_filters}
        dirs = [root / name for name in sorted(wanted)]
        return [d for d in dirs if d.is_dir()]
    return sorted([p for p in root.glob("shot_count_*") if p.is_dir()])


def collect_samples(
    assets_root: Path,
    shot_count_filters: list[str],
    sample_names: list[str],
    sample_pattern: str,
) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    shot_count_dirs = list_shot_count_dirs(assets_root, shot_count_filters)

    for shot_count_dir in shot_count_dirs:
        shot_count_name = shot_count_dir.name
        if sample_names:
            sample_dirs = [shot_count_dir / name for name in sample_names]
        else:
            sample_dirs = sorted([p for p in shot_count_dir.glob(sample_pattern) if p.is_dir()])
        for sample_dir in sample_dirs:
            if sample_dir.is_dir():
                samples.append((shot_count_name, sample_dir))
    return samples


def select_samples(
    samples: list[tuple[str, Path]],
    sample_limit: int,
    sample_select: str,
) -> list[tuple[str, Path]]:
    if sample_limit <= 0 or len(samples) <= sample_limit:
        return samples
    if sample_select == "random":
        copied = list(samples)
        import random
        random.shuffle(copied)
        return copied[:sample_limit]
    return samples[:sample_limit]


def collect_shot_tasks_from_samples(samples: list[tuple[str, Path]]) -> list[ShotTask]:
    tasks: list[ShotTask] = []
    for shot_count_name, sample_dir in samples:
        shots_dir = sample_dir / "shots"
        if not shots_dir.is_dir():
            continue
        for shot_dir in sorted([p for p in shots_dir.glob("shot_*") if p.is_dir()]):
            first_frame = shot_dir / "first_frame.jpg"
            gen_result = shot_dir / "generation_result.json"
            if not first_frame.is_file():
                continue
            tasks.append(
                ShotTask(
                    shot_count=shot_count_name,
                    sample_name=sample_dir.name,
                    shot_name=shot_dir.name,
                    first_frame_path=first_frame,
                    generation_result_path=gen_result,
                )
            )
    return tasks


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_sample_index_from_sample_name(sample_name: str) -> int | None:
    m = re.match(r"sample_(\d+)", sample_name)
    if not m:
        return None
    return int(m.group(1))


def parse_shot_index_from_shot_name(shot_name: str) -> int | None:
    m = re.match(r"shot_(\d+)$", shot_name)
    if not m:
        return None
    return int(m.group(1))


def shot_count_to_number(shot_count_name: str) -> str:
    m = re.search(r"(\d+)$", shot_count_name)
    if not m:
        raise ValueError(f"Cannot parse shot count number from {shot_count_name}")
    return m.group(1)


def resolve_prompt_source_dir(shot_count_name: str, prompt_source_template: str) -> Path:
    count = shot_count_to_number(shot_count_name)
    return Path(prompt_source_template.format(count=count)).expanduser().resolve()


def build_prompt_index_for_dir(prompt_source_dir: Path) -> dict[int, dict[int, PromptEntry]]:
    if not prompt_source_dir.is_dir():
        raise FileNotFoundError(f"Prompt source dir not found: {prompt_source_dir}")

    sample_map: dict[int, dict[int, PromptEntry]] = {}
    for sample_path in sorted(prompt_source_dir.glob("sample_*.json")):
        data = read_json(sample_path)
        sample_index = data.get("sample_index")
        if sample_index is None:
            m = re.match(r"sample_(\d+)\.json$", sample_path.name)
            if not m:
                continue
            sample_index = int(m.group(1))
        sample_idx = int(sample_index)
        shot_map = sample_map.setdefault(sample_idx, {})
        for shot in data.get("shots", []):
            if not isinstance(shot, dict):
                continue
            shot_index = shot.get("shot_index")
            prompt = str(shot.get("prompt", "")).strip()
            if shot_index is None or not prompt:
                continue
            shot_idx = int(shot_index)
            shot_map[shot_idx] = PromptEntry(
                prompt=prompt,
                duration_raw=shot.get("duration"),
                source_file=str(sample_path),
            )
    return sample_map


def normalize_duration_to_enum(duration_raw: Any, fallback_duration: str) -> str:
    if duration_raw is None:
        return str(fallback_duration)
    try:
        duration_num = float(duration_raw)
    except (TypeError, ValueError):
        return str(fallback_duration)

    # RunningHub expects integer enum strings [1..16].
    duration_int = int(math.floor(duration_num + 0.5))
    duration_int = max(1, min(16, duration_int))
    return str(duration_int)


def make_headers(api_key: str, content_type: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers=make_headers(api_key, "application/json"),
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc


def retry_delay_seconds(
    attempt: int,
    retry_initial_delay: float,
    retry_backoff: float,
    retry_max_delay: float,
) -> float:
    delay = retry_initial_delay * (retry_backoff ** max(0, attempt - 1))
    return max(0.0, min(delay, retry_max_delay))


def run_with_retry(
    op_name: str,
    shot_id: str,
    fn: Any,
    max_retries: int,
    retry_initial_delay: float,
    retry_backoff: float,
    retry_max_delay: float,
) -> Any:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # pylint: disable=broad-except
            if attempt >= max_retries:
                raise
            attempt += 1
            delay = retry_delay_seconds(
                attempt=attempt,
                retry_initial_delay=retry_initial_delay,
                retry_backoff=retry_backoff,
                retry_max_delay=retry_max_delay,
            )
            print(f"[Retry] {shot_id} {op_name} failed: {exc} | retry {attempt}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)


def encode_multipart(file_path: Path, field_name: str = "file") -> tuple[bytes, str]:
    boundary = f"----RunningHubBoundary{int(time.time() * 1000)}"
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'.encode(
            "utf-8"
        ),
        f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), boundary


def upload_image(upload_endpoint: str, api_key: str, image_path: Path) -> str:
    body, boundary = encode_multipart(image_path)
    request = urllib.request.Request(
        url=upload_endpoint,
        method="POST",
        data=body,
        headers=make_headers(api_key, f"multipart/form-data; boundary={boundary}"),
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upload failed HTTP {exc.code}: {detail}") from exc

    code = payload.get("code")
    if code not in (0, "0", None):
        raise RuntimeError(f"Upload failed: {payload}")
    data = payload.get("data") or {}
    url = data.get("download_url")
    if not isinstance(url, str) or not url:
        raise RuntimeError(f"Upload response missing download_url: {payload}")
    return url


def query_task(query_endpoint: str, api_key: str, task_id: str) -> dict[str, Any]:
    return post_json(query_endpoint, api_key, {"taskId": task_id})


def poll_until_done(
    query_endpoint: str,
    api_key: str,
    task_id: str,
    poll_interval: int,
    timeout_seconds: int,
    max_retries: int,
    retry_initial_delay: float,
    retry_backoff: float,
    retry_max_delay: float,
    shot_id: str,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] = {}
    transient_errors = 0
    while time.time() < deadline:
        try:
            payload = query_task(query_endpoint, api_key, task_id)
            transient_errors = 0
        except Exception as exc:  # pylint: disable=broad-except
            if transient_errors >= max_retries:
                raise RuntimeError(
                    f"Polling query failed too many times for task {task_id}: {exc}"
                ) from exc
            transient_errors += 1
            delay = retry_delay_seconds(
                attempt=transient_errors,
                retry_initial_delay=retry_initial_delay,
                retry_backoff=retry_backoff,
                retry_max_delay=retry_max_delay,
            )
            print(
                f"[Retry] {shot_id} poll query failed: {exc} | retry {transient_errors}/{max_retries} in {delay:.1f}s"
            )
            time.sleep(delay)
            continue
        last_payload = payload
        status = str(payload.get("status", "")).upper()
        if status in SUCCESS_STATES or status in FAILED_STATES:
            return payload
        if status in RUNNING_STATES or not status:
            time.sleep(max(1, poll_interval))
            continue
        time.sleep(max(1, poll_interval))
    raise TimeoutError(f"Polling timed out for task {task_id}. Last payload: {last_payload}")


def pick_result_url(payload: dict[str, Any]) -> str | None:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    for item in results:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def download_to(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response:
            target.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Download failed HTTP {exc.code} for {url}: {detail}") from exc


def ensure_success_record(record_path: Path) -> bool:
    if not record_path.is_file():
        return False
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("final_status", "")).upper() in SUCCESS_STATES


def resolve_prompt_entry_for_task(
    task: ShotTask,
    prompt_index_by_shot_count: dict[str, dict[int, dict[int, PromptEntry]]],
) -> tuple[PromptEntry, int, int]:
    sample_idx = parse_sample_index_from_sample_name(task.sample_name)
    if sample_idx is None:
        raise RuntimeError(f"Cannot parse sample index from {task.sample_name}")
    shot_idx = parse_shot_index_from_shot_name(task.shot_name)
    if shot_idx is None:
        raise RuntimeError(f"Cannot parse shot index from {task.shot_name}")

    by_sample = prompt_index_by_shot_count.get(task.shot_count)
    if by_sample is None:
        raise RuntimeError(f"No prompt source index loaded for {task.shot_count}")
    by_shot = by_sample.get(sample_idx)
    if by_shot is None:
        raise RuntimeError(
            f"No prompt source sample found for {task.shot_count}/{task.sample_name} "
            f"(sample_index={sample_idx})"
        )
    entry = by_shot.get(shot_idx)
    if entry is None:
        raise RuntimeError(
            f"No prompt source shot found for {task.shot_id} "
            f"(sample_index={sample_idx}, shot_index={shot_idx})"
        )
    return entry, sample_idx, shot_idx


def process_one_shot(
    task: ShotTask,
    api_key: str,
    output_dir: Path,
    prompt_index_by_shot_count: dict[str, dict[int, dict[int, PromptEntry]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    record_dir = output_dir / task.shot_count / task.sample_name / task.shot_name
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / "result.json"
    prompt_preview_path = record_dir / "prompt.txt"

    if args.skip_existing_success and ensure_success_record(record_path):
        return {
            "shot_id": task.shot_id,
            "state": "SKIPPED_SUCCESS",
            "record_path": str(record_path),
        }

    prompt_entry, sample_idx, shot_idx = resolve_prompt_entry_for_task(task, prompt_index_by_shot_count)
    video_prompt = prompt_entry.prompt.strip()
    if not video_prompt:
        raise RuntimeError(f"Prompt is empty for {task.shot_id}")
    resolved_duration = normalize_duration_to_enum(prompt_entry.duration_raw, args.duration)
    prompt_preview_path.write_text(video_prompt + "\n", encoding="utf-8")

    if args.dry_run:
        dry = {
            "shot_id": task.shot_id,
            "state": "DRY_RUN",
            "video_prompt": video_prompt,
            "sample_index": sample_idx,
            "shot_index": shot_idx,
            "duration_raw": prompt_entry.duration_raw,
            "resolved_duration": resolved_duration,
            "prompt_source_file": prompt_entry.source_file,
            "first_frame_path": str(task.first_frame_path),
        }
        record_path.write_text(json.dumps(dry, ensure_ascii=False, indent=2), encoding="utf-8")
        return dry

    uploaded_url = run_with_retry(
        op_name="upload",
        shot_id=task.shot_id,
        fn=lambda: upload_image(args.upload_endpoint, api_key, task.first_frame_path),
        max_retries=args.max_retries,
        retry_initial_delay=args.retry_initial_delay,
        retry_backoff=args.retry_backoff,
        retry_max_delay=args.retry_max_delay,
    )
    submit_payload = {
        "prompt": video_prompt,
        "imageUrl": uploaded_url,
        "duration": resolved_duration,
        "resolution": args.resolution,
        "audio": bool(args.audio),
    }
    submit_response = run_with_retry(
        op_name="submit",
        shot_id=task.shot_id,
        fn=lambda: post_json(args.generate_endpoint, api_key, submit_payload),
        max_retries=args.max_retries,
        retry_initial_delay=args.retry_initial_delay,
        retry_backoff=args.retry_backoff,
        retry_max_delay=args.retry_max_delay,
    )
    task_id = str(submit_response.get("taskId", "")).strip()
    if not task_id:
        raise RuntimeError(f"Submit response missing taskId: {submit_response}")

    final_payload = poll_until_done(
        query_endpoint=args.query_endpoint,
        api_key=api_key,
        task_id=task_id,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_initial_delay=args.retry_initial_delay,
        retry_backoff=args.retry_backoff,
        retry_max_delay=args.retry_max_delay,
        shot_id=task.shot_id,
    )
    final_status = str(final_payload.get("status", "")).upper()
    result_url = pick_result_url(final_payload)
    downloaded_path = None

    if args.download and final_status in SUCCESS_STATES and result_url:
        output_file = record_dir / "video.mp4"
        run_with_retry(
            op_name="download",
            shot_id=task.shot_id,
            fn=lambda: download_to(result_url, output_file),
            max_retries=args.max_retries,
            retry_initial_delay=args.retry_initial_delay,
            retry_backoff=args.retry_backoff,
            retry_max_delay=args.retry_max_delay,
        )
        downloaded_path = str(output_file)

    result_payload = {
        "shot_id": task.shot_id,
        "state": "DONE",
        "sample_index": sample_idx,
        "shot_index": shot_idx,
        "duration_raw": prompt_entry.duration_raw,
        "resolved_duration": resolved_duration,
        "prompt_source_file": prompt_entry.source_file,
        "task_id": task_id,
        "uploaded_image_url": uploaded_url,
        "submit_response": submit_response,
        "final_status": final_status,
        "final_response": final_payload,
        "result_url": result_url,
        "downloaded_path": downloaded_path,
        "prompt_path": str(prompt_preview_path),
    }
    record_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_payload


def main() -> None:
    args = parse_args()
    api_key = get_runninghub_api_key()
    if not args.dry_run and not api_key:
        raise RuntimeError("RUNNINGHUB_API_KEY is required unless --dry-run is set. Please set it in config.py.")

    raw_assets_root = Path(args.assets_root).expanduser()
    candidates = [raw_assets_root]
    if args.assets_root == "runninghub_outputs 2":
        candidates.append(Path("runninghub2"))
    if args.assets_root == "runninghub2":
        candidates.append(Path("runninghub_outputs 2"))

    assets_root = None
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            assets_root = resolved
            break

    if assets_root is None:
        assets_root = raw_assets_root.resolve()
    if not assets_root.is_dir():
        raise FileNotFoundError(f"assets root not found: {assets_root}")

    shot_counts = parse_csv(args.shot_counts)
    sample_names = parse_csv(args.sample_names)
    samples = collect_samples(
        assets_root=assets_root,
        shot_count_filters=shot_counts,
        sample_names=sample_names,
        sample_pattern=args.sample_pattern,
    )
    samples = select_samples(
        samples=samples,
        sample_limit=args.sample_limit,
        sample_select=args.sample_select,
    )
    tasks = collect_shot_tasks_from_samples(samples)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    if not tasks:
        raise RuntimeError("No shot tasks found after filters.")

    prompt_source_dirs: dict[str, str] = {}
    prompt_index_by_shot_count: dict[str, dict[int, dict[int, PromptEntry]]] = {}
    for shot_count_name in sorted({task.shot_count for task in tasks}):
        prompt_source_dir = resolve_prompt_source_dir(shot_count_name, args.prompt_source_template)
        prompt_source_dirs[shot_count_name] = str(prompt_source_dir)
        prompt_index_by_shot_count[shot_count_name] = build_prompt_index_for_dir(prompt_source_dir)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"[Info] Found {len(tasks)} shots. Concurrency={args.concurrency}, dry_run={args.dry_run}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_to_task = {
            executor.submit(process_one_shot, task, api_key, output_dir, prompt_index_by_shot_count, args): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                summary.append(result)
                print(f"[OK] {task.shot_id} -> {result.get('state')} ({result.get('final_status', '')})")
            except Exception as exc:  # pylint: disable=broad-except
                err = {
                    "shot_id": task.shot_id,
                    "error": str(exc),
                }
                errors.append(err)
                print(f"[ERR] {task.shot_id} -> {exc}")

    summary_payload = {
        "assets_root": str(assets_root),
        "prompt_source_template": args.prompt_source_template,
        "prompt_source_dirs": prompt_source_dirs,
        "selected_sample_count": len(samples),
        "selected_samples": [f"{shot_count}/{sample_dir.name}" for shot_count, sample_dir in samples],
        "total": len(tasks),
        "success_count": len(summary),
        "error_count": len(errors),
        "results": summary,
        "errors": errors,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Summary written: {summary_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
