#!/bin/bash
set -e

echo "[1/6] 生成分镜帧、视频以及时间戳..."
python extract_frames.py

echo "[2/6] 生成人物表和剧情..."
python video_scene_analyzer.py

echo "[3/6] Transcribe 分镜台词..."
python transcribe.py saved

echo "[4/6] Fix transcript..."
python fix_transcripts.py saved

echo "[5/6] 生成分镜描述..."
python image_shot_analyzer.py saved

echo "[6/6] Check 并合并数据..."
python check.py saved

echo ""
echo "全部完成！"
