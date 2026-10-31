# 人脸检测与表情分析

作者 medejavu-cmyk 独立完成，含 AI 辅助。基于 YOLO 的本地 Tkinter 界面，支持人脸检测、表情分类与结果展示；源码含可选模型接口。

## 本地运行

在项目目录使用 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe face_detect_gui.py
```

通过界面分别选择人脸检测及表情分类权重。原文件名为 `yolov11n-face.pt` 和 `yolo11s-emotion.pt`；本候选不再分发权重，来源及许可需补齐，因此不是开箱即用版本。训练脚本中的路径也需按本机数据位置调整。

可选大模型功能读取 `SILICONFLOW_API_KEY`，不要提交真实密钥。调用该功能可能发送图片到配置的外部服务，应仅使用有授权的素材。

## 来源与状态

依赖 Ultralytics、OpenCV、Pillow、NumPy；不包含原目录内的 Ultralytics 副本。第三方库与权重分别遵循其许可。本候选尚未完成运行复现，项目许可证待结合第三方依赖确认。仅做本地学习演示。
