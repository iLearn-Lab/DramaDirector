# Preprocess Pipeline

This directory contains the preprocessing pipeline for short-drama and video data. It converts raw episodes into keyframes, plot summaries, transcripts, structured shot descriptions, and multimodal embeddings.

## Local Configuration

Edit [config.py](/Users/mac/.dev/DramaDirector/config.py) directly before running the preprocessing pipeline.

The most important fields are:

```text
DASHSCOPE_API_KEY
DASHSCOPE_API_KEYS
SAVED_DIR
PROCESSED_SPLIT_DIR
TEXT_EMB_DIR
DEPTH_EMB_DIR
POSE_EMB_DIR
```

## Default Paths

The repository centralizes path defaults through [config.py](/Users/mac/.dev/DramaDirector/config.py) and [project_paths.py](/Users/mac/.dev/DramaDirector/project_paths.py). Unless you edit them, preprocessing will read from and write to:

```text
preprocess/saved/
preprocess/processed_split/
preprocess/text_emb/
preprocess/depth_emb/
preprocess/pose_emb/
```

## Input Layout

It is recommended to organize raw videos as `videos/<drama_name>/`, for example:

```text
videos/
  drama_a/
    1.mp4
    2.mp4
  drama_b/
    1.mp4
```

The pipeline then generates or consumes the following structure:

```text
preprocess/saved/<drama>/split/<episode>/segments_timeline.json   # shot timeline and keyframes
preprocess/saved/<drama>/plot/<episode>.json                      # characters, plot summary, global dialogues
preprocess/saved/<drama>/transcript/<episode>.json                # raw ASR transcript
preprocess/saved/<drama>/transcript/fixed_<episode>.json          # corrected transcript
preprocess/saved/<drama>/shot/<episode>.json                      # structured shot descriptions
preprocess/processed_split/<drama>/<episode>/{origin,depth,pose}/ # rendered frame assets
preprocess/{text_emb,depth_emb,pose_emb}/                         # embedding outputs
```

## Recommended Workflow

Run the following commands from the `preprocess/` directory:

```bash
python extract_frames.py videos
python video_scene_analyzer.py videos
python transcribe.py saved
python fix_transcripts.py saved
python image_shot_analyzer.py saved
python check.py saved
python process_pipeline.py
python embed_texts.py
python embed_images.py
```

You can also execute the main preprocessing chain with:

```bash
bash prepare_dataset.sh
```

## Depth and Pose Dependencies

`process_pipeline.py` generates `origin`, `depth`, and `pose` assets from saved keyframes. Before running it, clone the following projects into the current directory:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
git clone https://github.com/IDEA-Research/DWPose.git
```

The resulting directory layout should look like:

```text
Depth-Anything-V2/
  checkpoints/
DWPose/
  ControlNet-v1-1-nightly/
```

Depth-Anything-V2 also requires the corresponding checkpoint weights, for example:

```text
Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
```

Once the dependencies are ready, run:

```bash
python process_pipeline.py
```

## Script Summary

- `extract_frames.py`: scans `videos/<drama>/`, uses `shot_split.ShotSplitter`, and generates keyframes plus `segments_timeline.json`.
- `video_scene_analyzer.py`: calls a DashScope multimodal model to analyze each episode and outputs `plot/<episode>.json`.
- `transcribe.py`: reads the shot timeline, extracts audio with `ffmpeg`, and runs the local FunASR model to produce `transcript/<episode>.json`.
- `fix_transcripts.py`: uses plot information and reference dialogue to correct ASR errors and writes `fixed_<episode>.json`.
- `image_shot_analyzer.py`: combines keyframes, plot information, corrected transcripts, and durations to generate structured shot descriptions in `shot/<episode>.json`.
- `check.py`: validates whether `shot`, `transcript`, and `split` have consistent shot counts, and backfills dialogue and duration into shot files; errors are written to `error.json`.
- `process_pipeline.py`: converts saved keyframes into `origin/depth/pose` assets under `processed_split/`, using local `Depth-Anything-V2` and `DWPose`.
- `embed_texts.py`: generates `text_emb` from shot descriptions.
- `embed_images.py`: reads `processed_split/<drama>/<episode>/depth` and `pose`, then generates `depth_emb` and `pose_emb`.

## Embedding

Text embeddings:

```bash
python embed_texts.py
```

Image embeddings:

```bash
python embed_images.py
```

If you prefer to override the key from the command line instead of editing `config.py`:

```bash
python embed_texts.py --api_key "your_api_key"
python embed_images.py --api_key "your_api_key"
```

If you are still using the legacy flat layout `depth/` and `pose/`, enable compatibility mode:

```bash
python embed_images.py --source_root . --legacy_flat
```

## Notes

- Scripts that require DashScope will fail fast if `DASHSCOPE_API_KEY` is left empty in `config.py`.
- `transcribe.py` depends on the local `Fun-ASR/FunAudioLLM/Fun-ASR-Nano-2512` model directory.
- Before generating depth and pose assets, you must first clone `Depth-Anything-V2` and `DWPose` and prepare the Depth-Anything-V2 checkpoint weights.
- `check.py` may delete inconsistent `fixed_<episode>.json` or `shot/<episode>.json` files in order to force a clean rerun of failed cases.
