import argparse
import json
import re
from pathlib import Path
from typing import Any


END_TOKEN = "<END>"

SHOT_KEYS = [
    "index",
    "shot_scale",
    "camera_angle",
    "camera_motion",
    "subjects",
    "background",
    "description_narrative",
    "dialogue",
    "speaker",
    "emotion",
    "duration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean infer_results JSON files and export video-ready storyboard inputs."
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        default="infer_results*.json",
        help="Glob for infer_results JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prepared_video_inputs",
        help="Output root directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when a bad model_output is encountered.",
    )
    return parser.parse_args()


def extract_section(text: str, title: str) -> str:
    pattern = rf"【{re.escape(title)}】\n(.*?)(?=\n【|\Z)"
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else ""


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def unwrap_model_output(text: str) -> str:
    cleaned = strip_code_fence(text).replace(END_TOKEN, "").strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        try:
            decoded = json.loads(cleaned)
            if isinstance(decoded, str):
                cleaned = decoded.strip()
        except json.JSONDecodeError:
            pass
    return cleaned.strip()


def extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = unwrap_model_output(text)

    parse_candidates = [cleaned]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end >= start:
        parse_candidates.append(cleaned[start : end + 1])

    for candidate in parse_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse model_output into a JSON array.")


def normalize_subject(subject: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": subject.get("name"),
        "gender": subject.get("gender"),
        "clothing": subject.get("clothing"),
        "position": subject.get("position"),
        "action": subject.get("action"),
        "expression": subject.get("expression"),
    }


def normalize_shot(shot: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {key: shot.get(key) for key in SHOT_KEYS}
    normalized["subjects"] = [
        normalize_subject(subj) for subj in (shot.get("subjects") or []) if isinstance(subj, dict)
    ]

    try:
        normalized["duration"] = float(normalized["duration"]) if normalized["duration"] is not None else None
    except (TypeError, ValueError):
        normalized["duration"] = None

    return normalized


def format_subject(subject: dict[str, Any]) -> str:
    parts = []
    if subject.get("name"):
        parts.append(str(subject["name"]))
    if subject.get("position"):
        parts.append(f"位于{subject['position']}")
    if subject.get("clothing"):
        parts.append(f"服装：{subject['clothing']}")
    if subject.get("action"):
        parts.append(f"动作：{subject['action']}")
    if subject.get("expression"):
        parts.append(f"表情：{subject['expression']}")
    return "，".join(parts)


def build_shot_prompt(shot: dict[str, Any]) -> str:
    parts = []
    if shot.get("shot_scale") or shot.get("camera_angle") or shot.get("camera_motion"):
        parts.append(
            f"{shot.get('shot_scale') or ''}，{shot.get('camera_angle') or ''}，{shot.get('camera_motion') or ''}".strip("，")
        )
    if shot.get("background"):
        parts.append(f"背景：{shot['background']}")

    subjects = [format_subject(subj) for subj in shot.get("subjects", []) if format_subject(subj)]
    if subjects:
        parts.append("人物：" + "；".join(subjects))

    if shot.get("description_narrative"):
        parts.append(f"画面描述：{shot['description_narrative']}")
    if shot.get("dialogue"):
        speaker = shot.get("speaker") or "角色"
        parts.append(f"台词：{speaker}说“{shot['dialogue']}”")
    if shot.get("emotion"):
        parts.append(f"情绪：{shot['emotion']}")

    return "。".join(part for part in parts if part).strip("。") + "。"


def build_text_export(
    source_name: str,
    sample_index: int,
    summary: str,
    characters: str,
    shots: list[dict[str, Any]],
) -> str:
    lines = [
        f"Source File: {source_name}",
        f"Sample Index: {sample_index}",
        "",
        "Story Summary:",
        summary or "(empty)",
        "",
        "Characters:",
        characters or "(empty)",
        "",
        "Shot Prompts:",
    ]

    for shot in shots:
        duration = f"{shot['duration']:.1f}s" if isinstance(shot.get("duration"), float) else "unknown"
        lines.append(f"- Shot {shot.get('index')} | {duration}")
        lines.append(build_shot_prompt(shot))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def process_file(
    input_path: Path,
    output_dir: Path,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    manifest_entries = []
    error_entries = []
    source_stem = input_path.stem

    for record in records:
        sample_index = int(record["sample_index"])
        user_text = record.get("user", "")
        summary = extract_section(user_text, "剧情摘要")
        characters = extract_section(user_text, "出场角色")
        try:
            shots = [normalize_shot(shot) for shot in extract_json_array(record.get("model_output", ""))]
        except Exception as e:
            error_entries.append(
                {
                    "source_file": str(input_path),
                    "sample_index": sample_index,
                    "error": f"{type(e).__name__}: {e}",
                    "model_output_preview": str(record.get("model_output", ""))[:1000],
                }
            )
            if strict:
                raise
            print(f"[WARN] Skipping {input_path.name} sample_index={sample_index}: {e}")
            continue

        storyboard_path = output_dir / "storyboards" / source_stem / f"sample_{sample_index:03d}.json"
        video_input_path = output_dir / "video_inputs" / source_stem / f"sample_{sample_index:03d}.json"
        prompt_text_path = output_dir / "prompt_text" / source_stem / f"sample_{sample_index:03d}.txt"

        video_input = {
            "source_file": str(input_path),
            "sample_index": sample_index,
            "story_summary": summary,
            "characters": characters,
            "user_prompt": user_text,
            "shots": [
                {
                    "shot_index": shot.get("index"),
                    "duration": shot.get("duration"),
                    "prompt": build_shot_prompt(shot),
                    "storyboard": shot,
                }
                for shot in shots
            ],
        }

        write_json(storyboard_path, shots)
        write_json(video_input_path, video_input)
        write_text(prompt_text_path, build_text_export(source_stem, sample_index, summary, characters, shots))

        manifest_entries.append(
            {
                "source_file": str(input_path),
                "sample_index": sample_index,
                "storyboard_path": str(storyboard_path),
                "video_input_path": str(video_input_path),
                "prompt_text_path": str(prompt_text_path),
                "num_shots": len(shots),
            }
        )

    return manifest_entries, error_entries


def main() -> None:
    args = parse_args()
    root = Path(".")
    output_dir = Path(args.output_dir)
    input_paths = sorted(root.glob(args.input_glob))

    if not input_paths:
        raise FileNotFoundError(f"No files matched input_glob={args.input_glob!r}")

    all_entries = []
    all_errors = []
    for input_path in input_paths:
        entries, errors = process_file(input_path, output_dir, strict=args.strict)
        all_entries.extend(entries)
        all_errors.extend(errors)

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    errors_path = output_dir / "errors.jsonl"
    with open(errors_path, "w", encoding="utf-8") as f:
        for entry in all_errors:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Processed {len(input_paths)} infer_results files.")
    print(f"Exported {len(all_entries)} samples to {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    if all_errors:
        print(f"Skipped {len(all_errors)} bad samples. Details saved to: {errors_path}")
    else:
        print("No bad samples were skipped.")


if __name__ == "__main__":
    main()
