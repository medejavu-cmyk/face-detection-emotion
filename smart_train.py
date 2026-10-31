# smart_train_or_copy.py
"""
智能处理表情模型：
- 如果 train/weights/best.pt 存在 → 直接复制重命名（快速模式）
- 否则 → 自动训练并生成（完整模式）
"""

import os
import shutil
from ultralytics import YOLO

# ==================== 配置区 ====================
# 数据集路径
DATA_PATH = r"D:\深度学习人脸识别\人脸分类数据集"

# 已有训练结果路径
EXISTING_WEIGHTS_DIR = r"D:\深度学习人脸识别\train\weights"
BEST_PT_PATH = os.path.join(EXISTING_WEIGHTS_DIR, "best.pt")

# 目标输出路径（项目根目录）
PROJECT_ROOT = r"D:\深度学习人脸识别\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型"
FINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo11s-emotion.pt")

# 训练参数（仅当需要训练时使用）
TRAIN_CONFIG = {
    "model": "yolov8s-cls.pt",
    "epochs": 100,
    "imgsz": 224,
    "batch": 16,
    "name": "emotion_train_new"
}

# ==================== 主逻辑 ====================
def main():
    print("🔍 检查现有模型...")
    
    # 模式1: 如果 best.pt 已存在，直接复制（推荐！省时）
    if os.path.exists(BEST_PT_PATH):
        print("✅ 发现已训练的 best.pt，正在复制重命名...")
        shutil.copy2(BEST_PT_PATH, FINAL_MODEL_PATH)
        print(f"🎉 成功！模型已保存至:\n{FINAL_MODEL_PATH}")
        
    # 模式2: 如果不存在，则自动训练
    else:
        print("⚠️ 未找到 best.pt，开始自动训练...")
        try:
            model = YOLO(TRAIN_CONFIG["model"])
            results = model.train(
                data=DATA_PATH,
                epochs=TRAIN_CONFIG["epochs"],
                imgsz=TRAIN_CONFIG["imgsz"],
                batch=TRAIN_CONFIG["batch"],
                name=TRAIN_CONFIG["name"],
                project="runs"
            )
            
            # 获取新训练的 best.pt
            new_best_pt = os.path.join("runs", "classify", TRAIN_CONFIG["name"], "weights", "best.pt")
            if os.path.exists(new_best_pt):
                shutil.copy2(new_best_pt, FINAL_MODEL_PATH)
                print(f"🎉 训练完成！模型已保存至:\n{FINAL_MODEL_PATH}")
            else:
                raise FileNotFoundError("训练未生成 best.pt")
                
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            return
    
    # 验证文件是否可加载
    try:
        _ = YOLO(FINAL_MODEL_PATH)
        print("\n✅ 模型验证通过！可直接用于 GUI 程序。")
    except Exception as e:
        print(f"\n⚠️ 模型加载警告: {e}")

if __name__ == "__main__":
    main()