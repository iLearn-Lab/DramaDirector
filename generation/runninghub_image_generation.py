from __future__ import annotations

"""Generate character reference images and shot first frames with RunningHub.

The script expects prepared sample folders under ``prepared_data`` by default:

    prepared_data/shot_count_5/sample_xxx/
    prepared_data/shot_count_6/sample_xxx/

Set ``RUNNINGHUB_API_KEY`` in the environment, or pass ``--api-key`` explicitly.
Use ``--dry-run`` to validate inputs without calling the API.
"""

import argparse
import concurrent.futures
import http.client
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib import error, request
import ssl


DEFAULT_API_BASE = "https://www.runninghub.cn/openapi/v2"
DEFAULT_UPLOAD_ENDPOINT = f"{DEFAULT_API_BASE}/media/upload/binary"
DEFAULT_TEXT_TO_IMAGE_ENDPOINT = f"{DEFAULT_API_BASE}/seedream-v5-lite/text-to-image"
DEFAULT_IMAGE_TO_IMAGE_ENDPOINT = f"{DEFAULT_API_BASE}/seedream-v5-lite/image-to-image"
DEFAULT_QUERY_ENDPOINT = f"{DEFAULT_API_BASE}/query"
DEFAULT_WIDTH = 1728
DEFAULT_HEIGHT = 3072
SUCCESS_STATES = {"SUCCESS"}
FAILED_STATES = {"FAILED"}


@dataclass
class GeneratedImage:
    name: str
    local_path: Path
    upload_url: str
    task_id: str


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path
    output_root: Path
    width: int
    height: int
    poll_interval: int
    timeout_seconds: int


class RunningHubClient:
    def __init__(
        self,
        api_key: str,
        upload_endpoint: str,
        text_to_image_endpoint: str,
        image_to_image_endpoint: str,
        query_endpoint: str,
        request_timeout: int,
        max_retries: int,
        retry_delay: float,
        retry_backoff: float,
        retry_max_delay: float,
    ):
        self.api_key = api_key
        self.upload_endpoint = upload_endpoint
        self.text_to_image_endpoint = text_to_image_endpoint
        self.image_to_image_endpoint = image_to_image_endpoint
        self.query_endpoint = query_endpoint
        self.request_timeout = request_timeout
        self.max_retries = max(0, max_retries)
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.retry_max_delay = retry_max_delay

    def _should_retry_exception(self, exc: Exception) -> bool:
        if isinstance(exc, error.URLError):
            return True
        if isinstance(exc, TimeoutError):
            return True
        if isinstance(exc, ssl.SSLError):
            return True
        if isinstance(exc, http.client.RemoteDisconnected):
            return True
        if isinstance(exc, ConnectionResetError):
            return True
        if isinstance(exc, ConnectionAbortedError):
            return True
        if isinstance(exc, ConnectionRefusedError):
            return True
        return False

    def _retry_request(self, func, description: str):
        last_exc = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                if not self._should_retry_exception(exc) or attempt > self.max_retries:
                    raise
                delay = min(self.retry_delay * (self.retry_backoff ** (attempt - 1)), self.retry_max_delay)
                print(
                    f"[retry] {description} failed ({attempt}/{self.max_retries}): {exc}; retry in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise last_exc

    def _json_request(self, url: str, payload: dict) -> dict:
        def _run():
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.request_timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP error {exc.code} for {url}: {detail}") from exc

        return self._retry_request(_run, f"json request {url}")

    def _build_multipart_body(self, file_path: Path, boundary: str) -> bytes:
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        file_bytes = file_path.read_bytes()
        parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        return b"".join(parts)

    def upload_file(self, file_path: Path) -> str:
        def _run():
            boundary = f"Boundary{int(time.time() * 1000)}"
            body = self._build_multipart_body(file_path, boundary)
            req = request.Request(
                self.upload_endpoint,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.request_timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP error {exc.code} for upload {file_path}: {detail}") from exc

        data = self._retry_request(_run, f"upload {file_path}")
        if data.get("code") not in (0, "0", None):
            raise RuntimeError(f"Upload failed for {file_path}: {data}")
        download_url = (data.get("data") or {}).get("download_url")
        if not download_url:
            raise RuntimeError(f"Upload response missing download_url for {file_path}: {data}")
        return download_url

    def submit_text_task(self, prompt: str, width: int, height: int) -> dict:
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "sequentialImageGeneration": "disabled",
            "maxImages": 1,
        }
        data = self._json_request(self.text_to_image_endpoint, payload)
        if data.get("errorCode") or str(data.get("status", "")).upper() in FAILED_STATES:
            raise RuntimeError(f"Text task submission failed: {data}")
        return data

    def submit_image_task(self, prompt: str, image_urls: list[str], width: int, height: int) -> dict:
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "imageUrls": image_urls,
            "sequentialImageGeneration": "disabled",
            "maxImages": 1,
        }
        data = self._json_request(self.image_to_image_endpoint, payload)
        if data.get("errorCode") or str(data.get("status", "")).upper() in FAILED_STATES:
            raise RuntimeError(f"Task submission failed: {data}")
        return data

    def query_task(self, task_id: str) -> dict:
        return self._json_request(self.query_endpoint, {"taskId": task_id})

    def wait_for_success(self, task_id: str, poll_interval: int, timeout_seconds: int) -> dict:
        deadline = time.time() + timeout_seconds
        last_data: dict[str, Any] = {}
        while time.time() < deadline:
            data = self.query_task(task_id)
            last_data = data
            status = str(data.get("status", "")).upper()
            if status in SUCCESS_STATES:
                return data
            if status in FAILED_STATES:
                raise RuntimeError(f"Task failed: {json.dumps(data, ensure_ascii=False)}")
            time.sleep(max(1, poll_interval))
        raise TimeoutError(f"Task {task_id} timed out after {timeout_seconds}s. Last response: {last_data}")

    def download_file(self, file_url: str, destination: Path) -> None:
        def _run():
            req = request.Request(
                file_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                method="GET",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with request.urlopen(req, timeout=self.request_timeout) as resp:
                    with destination.open("wb") as f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP error {exc.code} for download {file_url}: {detail}") from exc

        self._retry_request(_run, f"download {file_url}")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_") or "character"


def load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return read_json(path)


def build_character_prompt(base_prompt: str) -> str:
    return (
        f"{base_prompt}\n"
        "补充要求：输出竖版9:16单人角色参考图，人物完整入镜，服装和外观特征稳定，背景简洁干净，适合作为后续分镜角色一致性参考。"
    )


def strip_conflicting_control_lines(base_prompt: str) -> str:
    lines = []
    for raw_line in base_prompt.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(raw_line)
            continue
        if line.startswith("控制条件："):
            continue
        if line.startswith("上传顺序说明："):
            continue
        lines.append(raw_line)

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def strip_dialogue_content(base_prompt: str) -> str:
    cleaned = re.sub(r"台词：.*?(?=\n|$)", "", base_prompt)
    cleaned = re.sub(r"\s+。", "。", cleaned)
    cleaned = re.sub(r"。。+", "。", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_environment_only_prompt(base_prompt: str, character_refs: list[GeneratedImage]) -> bool:
    if character_refs:
        return False
    return "人物：" not in base_prompt


def build_shot_prompt(base_prompt: str, character_refs: list[GeneratedImage]) -> str:
    cleaned_base_prompt = strip_dialogue_content(strip_conflicting_control_lines(base_prompt))
    lines = [cleaned_base_prompt]
    environment_only = is_environment_only_prompt(cleaned_base_prompt, character_refs)

    if character_refs:
        lines.append("角色参考要求：")
        for idx, ref in enumerate(character_refs, start=1):
            lines.append(f"使用图{idx}作为角色参考图，保持角色 {ref.name} 的外观、脸部特征和服装一致。")

    depth_idx = len(character_refs) + 1
    pose_idx = len(character_refs) + 2
    lines.append(f"使用图{depth_idx}深度图作为空间结构参考，优先用于指导场景层次、透视关系和整体空间布局。")
    if environment_only:
        lines.append(f"使用图{pose_idx}骨架图仅作为轻度构图参考，不要据此生成任何人物、人体局部或人形轮廓。")
        lines.append("这是纯环境镜头，画面中不得出现任何人物；如果骨架图与文本描述冲突，必须以文本描述为主，忽略其中会引入人物的姿态信息。")
    else:
        lines.append(f"使用图{pose_idx}骨架图作为人物姿态参考，优先用于指导人物大体动作和肢体朝向。")
        lines.append("人物动作可以参考骨架图，但不能完全依赖骨架图；如果骨架图与文本描述存在不一致，以文本描述为主，仅吸收其中可用的动作和姿态信息，不要机械照搬控制图。")
        lines.append("人物动作要符合自然规律，姿态自然得体，肢体关系协调，不要过于僵硬、夸张或不自然。")
        lines.append("深度图虽然用于指导空间结构和透视关系，但不能因此破坏人物的正常尺度；如果深度图导致人物比例失衡或前后景人物尺寸差异过度，仍应以自然真实的人物比例为主。")
        lines.append("人物构图要求：两个人物处于同一真实场景空间中，人物比例必须正常，身高差与透视关系自然。")
        lines.append("角色一致性要求：人物的发色、发型与角色参考图保持一致，不要擅自改变头发颜色。")
    lines.append("画面中不要出现任何文字、字幕、对白气泡或其他可读文本。")
    lines.append("补充要求：输出竖版首帧图像，写实风格，主体清晰，构图稳定，整体画面表达以文本内容为最高优先级。")
    return "\n".join(lines)


def select_character_refs_for_shot(shot_metadata: dict, character_refs: list[GeneratedImage]) -> list[GeneratedImage]:
    search_text = "\n".join(
        [
            shot_metadata.get("shot_prompt", ""),
            shot_metadata.get("original_vlm_prompt", ""),
            shot_metadata.get("rewritten_vlm_prompt", ""),
        ]
    )
    matched = [ref for ref in character_refs if ref.name and ref.name in search_text]
    return matched


def choose_output_file(results: list[dict]) -> dict:
    for item in results:
        if item.get("url"):
            return item
    raise RuntimeError(f"No downloadable result found: {results}")


def maybe_load_existing_generated_image(output_dir: Path, name: str) -> Optional[GeneratedImage]:
    meta_path = output_dir / f"{sanitize_filename(name)}.json"
    if not meta_path.exists():
        return None
    data = read_json(meta_path)
    local_path = output_dir / data["file_name"]
    if not local_path.exists():
        return None
    return GeneratedImage(
        name=data["name"],
        local_path=local_path,
        upload_url=data["upload_url"],
        task_id=data["task_id"],
    )


def write_inflight_task(path: Path, data: dict) -> None:
    write_json(path, data)


def clear_inflight_task(path: Path) -> None:
    if path.exists():
        path.unlink()


def log_progress(message: str) -> None:
    print(message, flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def validate_sample_inputs(data_root: Path, sample_dir: Path) -> int:
    require_file(sample_dir / "characters" / "character_reference_prompts.json", "character prompts")
    manifest_path = sample_dir / "sample_manifest.json"
    require_file(manifest_path, "sample manifest")

    manifest = read_json(manifest_path)
    shots = manifest.get("shots")
    if not isinstance(shots, list):
        raise ValueError(f"sample_manifest.json must contain a shots list: {manifest_path}")

    for shot in shots:
        if not isinstance(shot, dict) or not shot.get("shot_dir"):
            raise ValueError(f"Invalid shot entry in {manifest_path}: {shot}")
        shot_dir = data_root / shot["shot_dir"]
        require_file(shot_dir / "metadata.json", "shot metadata")
        require_file(shot_dir / "prompt.txt", "shot prompt")
        require_file(shot_dir / "controls" / "01_depth.png", "depth control image")
        require_file(shot_dir / "controls" / "02_pose.png", "pose control image")
    return len(shots)


def generate_character_references(
    client: RunningHubClient,
    config: RuntimeConfig,
    sample_dir: Path,
    skip_existing: bool,
) -> list[GeneratedImage]:
    prompts_path = sample_dir / "characters" / "character_reference_prompts.json"
    prompts = read_json(prompts_path)
    output_dir = config.output_root / sample_dir.relative_to(config.data_root) / "characters"
    generated: list[GeneratedImage] = []

    for item in prompts:
        name = item["name"]
        existing = maybe_load_existing_generated_image(output_dir, name) if skip_existing else None
        if existing is not None:
            log_progress(f"  [character][skip] {name}")
            generated.append(existing)
            continue

        prompt = build_character_prompt(item["prompt"])
        inflight_path = output_dir / f"{sanitize_filename(name)}.inflight.json"
        inflight = load_json_if_exists(inflight_path) if skip_existing else None
        if inflight and inflight.get("task_id"):
            task_id = inflight["task_id"]
            log_progress(f"  [character][resume] {name} task={task_id}")
        else:
            log_progress(f"  [character][submit] {name}")
            submit = client.submit_text_task(prompt=prompt, width=config.width, height=config.height)
            task_id = submit["taskId"]
            write_inflight_task(
                inflight_path,
                {
                    "type": "character",
                    "name": name,
                    "prompt": prompt,
                    "task_id": task_id,
                    "selected_clothing": item.get("selected_clothing"),
                },
            )

        log_progress(f"  [character][wait] {name} task={task_id}")
        result = client.wait_for_success(
            task_id,
            poll_interval=config.poll_interval,
            timeout_seconds=config.timeout_seconds,
        )
        output = choose_output_file(result.get("results") or [])
        ext = output.get("outputType") or "png"
        file_stem = sanitize_filename(name)
        local_path = output_dir / f"{file_stem}.{ext}"
        client.download_file(output["url"], local_path)
        upload_url = client.upload_file(local_path)

        meta = {
            "name": name,
            "selected_clothing": item.get("selected_clothing"),
            "prompt": prompt,
            "task_id": task_id,
            "result_url": output["url"],
            "upload_url": upload_url,
            "file_name": local_path.name,
            "width": config.width,
            "height": config.height,
        }
        write_json(output_dir / f"{file_stem}.json", meta)
        clear_inflight_task(inflight_path)
        log_progress(f"  [character][done] {name} -> {local_path.name}")
        generated.append(
            GeneratedImage(
                name=name,
                local_path=local_path,
                upload_url=upload_url,
                task_id=task_id,
            )
        )

    return generated


def generate_shots(
    client: RunningHubClient,
    config: RuntimeConfig,
    sample_dir: Path,
    character_refs: list[GeneratedImage],
    skip_existing: bool,
    shot_concurrency: int,
) -> None:
    sample_manifest = read_json(sample_dir / "sample_manifest.json")
    total_shots = len(sample_manifest["shots"])

    def process_shot(shot: dict, shot_order: int) -> None:
        shot_dir = config.data_root / shot["shot_dir"]
        output_dir = config.output_root / Path(shot["shot_dir"])
        result_meta_path = output_dir / "generation_result.json"
        shot_label = f"{sample_manifest['sample_id']} shot={shot['shot_index']} ({shot_order}/{total_shots})"
        if skip_existing and result_meta_path.exists():
            log_progress(f"  [shot][skip] {shot_label}")
            return

        shot_metadata = read_json(shot_dir / "metadata.json")
        selected_character_refs = select_character_refs_for_shot(shot_metadata, character_refs)
        prompt = build_shot_prompt(read_text(shot_dir / "prompt.txt"), selected_character_refs)
        depth_path = shot_dir / "controls" / "01_depth.png"
        pose_path = shot_dir / "controls" / "02_pose.png"
        inflight_path = output_dir / "generation_inflight.json"

        inflight = load_json_if_exists(inflight_path) if skip_existing else None
        if inflight and inflight.get("task_id"):
            image_urls = inflight.get("submitted_image_urls", [])
            task_id = inflight["task_id"]
            log_progress(f"  [shot][resume] {shot_label} task={task_id}")
        else:
            log_progress(
                f"  [shot][submit] {shot_label} refs={len(selected_character_refs)} controls=2"
            )
            image_urls = [ref.upload_url for ref in selected_character_refs]
            image_urls.append(client.upload_file(depth_path))
            image_urls.append(client.upload_file(pose_path))

            submit = client.submit_image_task(
                prompt=prompt,
                image_urls=image_urls,
                width=config.width,
                height=config.height,
            )
            task_id = submit["taskId"]
            write_inflight_task(
                inflight_path,
                {
                    "type": "shot",
                    "sample_id": sample_manifest["sample_id"],
                    "shot_index": shot["shot_index"],
                    "task_id": task_id,
                    "prompt": prompt,
                    "submitted_image_urls": image_urls,
                    "character_reference_images": [
                        {"name": ref.name, "upload_url": ref.upload_url, "file": str(ref.local_path)}
                        for ref in selected_character_refs
                    ],
                    "control_images": [
                        {"type": "depth", "order": 1, "source": str(depth_path), "upload_url": image_urls[-2]},
                        {"type": "pose", "order": 2, "source": str(pose_path), "upload_url": image_urls[-1]},
                    ],
                },
            )

        log_progress(f"  [shot][wait] {shot_label} task={task_id}")
        result = client.wait_for_success(
            task_id,
            poll_interval=config.poll_interval,
            timeout_seconds=config.timeout_seconds,
        )
        output = choose_output_file(result.get("results") or [])
        ext = output.get("outputType") or "png"
        local_path = output_dir / f"first_frame.{ext}"
        client.download_file(output["url"], local_path)

        meta = {
            "sample_id": sample_manifest["sample_id"],
            "shot_index": shot["shot_index"],
            "task_id": task_id,
            "prompt": prompt,
            "character_reference_images": [
                {"name": ref.name, "upload_url": ref.upload_url, "file": str(ref.local_path)}
                for ref in selected_character_refs
            ],
            "control_images": [
                {"type": "depth", "order": 1, "source": str(depth_path), "upload_url": image_urls[-2]},
                {"type": "pose", "order": 2, "source": str(pose_path), "upload_url": image_urls[-1]},
            ],
            "submitted_image_urls": image_urls,
            "result_url": output["url"],
            "file_name": local_path.name,
            "width": config.width,
            "height": config.height,
        }
        write_json(result_meta_path, meta)
        clear_inflight_task(inflight_path)
        log_progress(f"  [shot][done] {shot_label} -> {local_path.name}")

    shots = sample_manifest["shots"]
    if shot_concurrency <= 1:
        for shot_order, shot in enumerate(shots, start=1):
            process_shot(shot, shot_order)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=shot_concurrency) as executor:
        futures = {
            executor.submit(process_shot, shot, shot_order): (shot, shot_order)
            for shot_order, shot in enumerate(shots, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
            shot, shot_order = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(
                    f"[shot-error] sample={sample_manifest['sample_id']} shot={shot['shot_index']} order={shot_order}/{total_shots} error={exc}",
                    file=sys.stderr,
                )
                raise


def iter_sample_dirs(data_root: Path, sample_id: Optional[str], shot_count: Optional[int]) -> list[Path]:
    roots = []
    if shot_count is None:
        roots = sorted(p for p in data_root.glob("shot_count_*") if p.is_dir())
    else:
        roots = [data_root / f"shot_count_{shot_count}"]

    sample_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if sample_id:
            target = root / sample_id
            if target.exists():
                sample_dirs.append(target)
            continue
        sample_dirs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return sample_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate character refs and first frames with RunningHub.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="prepared_data",
        help="Root directory containing shot_count_x/sample_xxx prepared inputs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runninghub_outputs",
        help="Root directory for generated images and metadata.",
    )
    parser.add_argument("--sample-id", help="Only process one sample, for example sample_010_s010.")
    parser.add_argument("--shot-count", type=int, help="Only process one shot count bucket, for example 5 or 6.")
    parser.add_argument("--sample-limit", type=int, default=0, help="Limit number of samples after filtering.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist.")
    parser.add_argument("--characters-only", action="store_true", help="Only generate character reference images.")
    parser.add_argument("--shots-only", action="store_true", help="Only generate shot first frames.")
    parser.add_argument("--api-key", default="", help="RunningHub API key. Defaults to RUNNINGHUB_API_KEY.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print selected samples; do not call API.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Generated image width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Generated image height.")
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel samples.")
    parser.add_argument("--shot-concurrency", type=int, default=2, help="Max parallel shots inside one sample.")
    parser.add_argument("--poll-interval", type=int, default=5, help="Task polling interval in seconds.")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Per-task polling timeout in seconds.")
    parser.add_argument("--request-timeout", type=int, default=120, help="HTTP request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for transient network errors.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Initial retry delay in seconds.")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Retry backoff multiplier.")
    parser.add_argument("--retry-max-delay", type=float, default=30.0, help="Maximum retry delay in seconds.")
    parser.add_argument("--upload-endpoint", default=DEFAULT_UPLOAD_ENDPOINT, help="RunningHub upload endpoint.")
    parser.add_argument(
        "--text-to-image-endpoint",
        default=DEFAULT_TEXT_TO_IMAGE_ENDPOINT,
        help="RunningHub text-to-image endpoint.",
    )
    parser.add_argument(
        "--image-to-image-endpoint",
        default=DEFAULT_IMAGE_TO_IMAGE_ENDPOINT,
        help="RunningHub image-to-image endpoint.",
    )
    parser.add_argument("--query-endpoint", default=DEFAULT_QUERY_ENDPOINT, help="RunningHub query endpoint.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = (args.api_key or os.environ.get("RUNNINGHUB_API_KEY", "")).strip()
    if not args.dry_run and not api_key:
        print(
            "Missing RunningHub API key. Set RUNNINGHUB_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1
    if args.characters_only and args.shots_only:
        print("--characters-only and --shots-only cannot be used together.", file=sys.stderr)
        return 1

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config = RuntimeConfig(
        data_root=data_root,
        output_root=output_root,
        width=args.width,
        height=args.height,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
    )
    if not config.data_root.is_dir():
        print(f"Data root not found: {config.data_root}", file=sys.stderr)
        return 1
    sample_dirs = iter_sample_dirs(data_root=config.data_root, sample_id=args.sample_id, shot_count=args.shot_count)
    if args.sample_limit > 0:
        sample_dirs = sample_dirs[: args.sample_limit]
    if not sample_dirs:
        print("No matching samples found.", file=sys.stderr)
        return 1

    print(
        f"[info] samples={len(sample_dirs)} data_root={config.data_root} "
        f"output_root={config.output_root} dry_run={args.dry_run}"
    )

    if args.dry_run:
        total_shots = 0
        for sample_dir in sample_dirs:
            shot_count = validate_sample_inputs(config.data_root, sample_dir)
            total_shots += shot_count
            print(f"[dry-run] {sample_dir.relative_to(config.data_root)} shots={shot_count}")
        print(f"[dry-run] validated samples={len(sample_dirs)} shots={total_shots}")
        return 0

    config.output_root.mkdir(parents=True, exist_ok=True)
    client = RunningHubClient(
        api_key=api_key,
        upload_endpoint=args.upload_endpoint,
        text_to_image_endpoint=args.text_to_image_endpoint,
        image_to_image_endpoint=args.image_to_image_endpoint,
        query_endpoint=args.query_endpoint,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        retry_backoff=args.retry_backoff,
        retry_max_delay=args.retry_max_delay,
    )

    def process_sample(sample_dir: Path) -> None:
        print(f"[sample] {sample_dir.relative_to(config.data_root)}")
        validate_sample_inputs(config.data_root, sample_dir)
        character_refs: list[GeneratedImage] = []

        if not args.shots_only:
            character_refs = generate_character_references(
                client=client,
                config=config,
                sample_dir=sample_dir,
                skip_existing=args.skip_existing,
            )
            print(f"  generated character refs: {len(character_refs)}")
        else:
            prompts = read_json(sample_dir / "characters" / "character_reference_prompts.json")
            output_dir = config.output_root / sample_dir.relative_to(config.data_root) / "characters"
            for item in prompts:
                existing = maybe_load_existing_generated_image(output_dir, item["name"])
                if existing is None:
                    raise RuntimeError(
                        f"Missing character reference output for {item['name']} in {sample_dir}. "
                        "Run without --shots-only first."
                    )
                character_refs.append(existing)
            print(f"  loaded character refs: {len(character_refs)}")

        if not args.characters_only:
            print(
                f"  generating shots: {len(read_json(sample_dir / 'sample_manifest.json')['shots'])} "
                f"with concurrency={args.concurrency}, shot_concurrency={args.shot_concurrency}"
            )
            generate_shots(
                client=client,
                config=config,
                sample_dir=sample_dir,
                character_refs=character_refs,
                skip_existing=args.skip_existing,
                shot_concurrency=max(1, args.shot_concurrency),
            )
            print("  generated shot first frames")

    if args.concurrency <= 1:
        for sample_dir in sample_dirs:
            process_sample(sample_dir)
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_sample, sample_dir): sample_dir for sample_dir in sample_dirs}
        for future in concurrent.futures.as_completed(futures):
            sample_dir = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[sample-error] sample={sample_dir.name} error={exc}", file=sys.stderr)
                raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
