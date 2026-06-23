import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DEFAULT_PROCESSED_SPLIT_DIR, DEFAULT_SAVED_DIR

# ==========================================
# 环境变量与模块导入
# ==========================================
sys.path.insert(0, str(Path(__file__).parent / "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

dwpose_path = Path(__file__).parent / "DWPose" / "ControlNet-v1-1-nightly"
sys.path.insert(0, str(dwpose_path))
try:
    from annotator.dwpose import DWposeDetector
except ImportError as e:
    print(f"⚠️ 导入 DWPose 失败: {e}")
    DWposeDetector = None

# ==========================================
# 1. 模型加载逻辑
# ==========================================
def load_depth_model(encoder="vitl", device="cuda"):
    print(f"📦 正在加载 Depth 模型 ({encoder}) 到 {device}...")
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    checkpoint_path = Path(__file__).parent / "Depth-Anything-V2" / "checkpoints" / f"depth_anything_v2_{encoder}.pth"
    if not checkpoint_path.exists():
        print(f"❌ 找不到深度模型权重: {checkpoint_path}")
        sys.exit(1)
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    return model.to(device).eval()

def load_pose_model(device="cuda"):
    print(f"📦 正在加载 DWPose 模型 到 {device}...")
    if DWposeDetector is None:
        print("❌ DWPose 模块未成功导入。")
        sys.exit(1)
    try:
        return DWposeDetector()
    except Exception as e:
        print(f"❌ DWPose 模型初始化失败: {e}")
        sys.exit(1)

# ==========================================
# 2. 核心推理函数 (GPU)
# ==========================================
@torch.no_grad()
def process_depth(model, raw_image, input_size=518):
    depth = model.infer_image(raw_image, input_size)
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0)
    return depth_normalized.astype(np.uint8)

@torch.no_grad()
def process_pose(model, raw_image):
    try:
        result = model(raw_image)
        return result[0] if isinstance(result, tuple) else result
    except:
        return np.zeros_like(raw_image)

# ==========================================
# 3. IO 读写逻辑 (修正后的原子写入)
# ==========================================
def save_task(origin_path, depth_path, pose_path, raw_image, depth_img, pose_img, display_path):
    """
    修正点：tmp 文件现在保持 .png 后缀，例如 tmp_1.png
    """
    for target_path, img_data in [
        (origin_path, raw_image),
        (depth_path, depth_img),
        (pose_path, pose_img)
    ]:
        if img_data is None: continue
        # 使用 with_name 确保临时文件也有 .png 后缀，让 OpenCV 认出格式
        tmp_path = target_path.with_name(f"tmp_{target_path.name}")
        if cv2.imwrite(str(tmp_path), img_data):
            os.replace(tmp_path, target_path)
    return display_path

def is_valid_file(p: Path):
    return p.exists() and p.stat().st_size > 0

# ==========================================
# 4. 主干调度逻辑
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    depth_model = load_depth_model(encoder="vitl", device=device)
    pose_model = load_pose_model(device=device)

    source_base = DEFAULT_SAVED_DIR
    target_base = DEFAULT_PROCESSED_SPLIT_DIR
    
    print("🔍 正在扫描文件目录...")
    tasks = []
    
    for drama_dir in source_base.iterdir():
        if not drama_dir.is_dir(): continue
        split_dir = drama_dir / "split"
        if not split_dir.exists(): continue
            
        for ep_dir in split_dir.iterdir():
            if not ep_dir.is_dir(): continue
            for img_path in ep_dir.glob("*.png"):
                out_dir = target_base / drama_dir.name / ep_dir.name
                ot, dt, pt = out_dir/"origin"/img_path.name, out_dir/"depth"/img_path.name, out_dir/"pose"/img_path.name
                
                if is_valid_file(ot) and is_valid_file(dt) and is_valid_file(pt):
                    continue
                
                (out_dir / "origin").mkdir(parents=True, exist_ok=True)
                (out_dir / "depth").mkdir(parents=True, exist_ok=True)
                (out_dir / "pose").mkdir(parents=True, exist_ok=True)
                
                # 保存相对路径用于日志显示
                display_rel_path = f"{drama_dir.name}/{ep_dir.name}/{img_path.name}"
                tasks.append((img_path, ot, dt, pt, display_rel_path))

    if not tasks:
        print("✅ 所有图片已处理完毕！")
        return

    print(f"🚀 开始处理 {len(tasks)} 张图片...")

    # 完成后的实时打印回调
    def on_complete(future):
        try:
            path_str = future.result()
            print(f"✅ [DONE] {path_str}")
        except Exception as e:
            print(f"❌ [FAIL] 保存失败: {e}")

    # 使用 12 线程并发写入硬盘
    with ThreadPoolExecutor(max_workers=12) as io_pool:
        for img_path, ot, dt, pt, display_path in tasks:
            raw_image = cv2.imread(str(img_path))
            if raw_image is None: continue
                
            # GPU 串行推理
            d_img = process_depth(depth_model, raw_image)
            p_img = process_pose(pose_model, raw_image)
            
            # 提交 IO 任务
            future = io_pool.submit(save_task, ot, dt, pt, raw_image, d_img, p_img, display_path)
            future.add_done_callback(on_complete)

    print("\n🎉 全部处理完成！")

if __name__ == "__main__":
    main()
