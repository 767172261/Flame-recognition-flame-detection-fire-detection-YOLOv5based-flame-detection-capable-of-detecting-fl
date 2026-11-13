火焰识别，火焰检测，火灾检测，基于yolov5的火焰检测，可以检测视频和图片，也可视频实时检测，检测准确率高，帧率很高，有标注好的训练数据集可以自己重新训练。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a6ce69321884fd38e75e268f1b4cce8.png)


可以将训练好的模型部署到英伟达边缘计算设备nano nx或实际场景中，如监控摄像头、无人机、消防设备等，实时检测火灾并发出警报，以提高消防安全性。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b2ca1e7d8d1f493499faf810a46bb38f.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7d89c4dd804d42f2b6c88b9a87d3de92.png)
以下是 **基于 YOLOv5 的高精度、高帧率火焰检测系统** 的完整解决方案，包含：

- ✅ 标注好的火焰数据集说明  
- ✅ 详细训练代码（支持自定义训练）  
- ✅ 视频/图片/摄像头实时检测  
- ✅ 模型导出与 **NVIDIA Jetson Nano/NX 部署脚本**  
- ✅ 报警触发逻辑（声音/日志/推流）  
- ✅ UI 可选（命令行 + OpenCV 可视化）

---

## 🔥 一、火焰检测数据集说明（可直接训练）

| 项目 | 内容 |
|------|------|
| **类别** | `fire`（火焰）、`smoke`（烟雾）——2 类（可扩展） |
| **图像数量** | 5,000+ 张（含白天/黑夜、室内/室外、不同火源） |
| **标注格式** | YOLO `.txt` 格式（LabelImg 标注） |
| **目录结构** |  
```
fire_dataset/
├── images/train/      # ~4000 张
├── images/val/
├── labels/train/
├── labels/val/
└── fire.yaml
```
| **fire.yaml** |
```yaml
train: ./images/train
val: ./images/val
nc: 2
names: ['fire', 'smoke']
```

> 💡 若你已有自己的数据集，只需按此结构组织即可。

---

## 🧪 二、环境准备（训练端）

```bash
# 克隆 YOLOv5 官方仓库
git clone https://github.com/ultralytics/yolov5
cd yolov5

# 安装依赖（Python ≥3.8）
pip install -r requirements.txt torch torchvision
```

---

## ▶️ 三、详细训练代码（YOLOv5）

### 1. 启动训练（命令行）

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data /path/to/fire_dataset/fire.yaml \
  --weights yolov5s.pt \
  --name fire_yolov5s \
  --cache ram \
  --workers 4
```

### 2. Python 脚本训练（`train_fire.py`）

```python
# train_fire.py
import torch
from yolov5 import train

if __name__ == '__main__':
    train.run(
        imgsz=640,
        batch_size=16,
        epochs=100,
        data='/path/to/fire_dataset/fire.yaml',
        weights='yolov5s.pt',
        name='fire_yolov5s',
        cache='ram',
        workers=4,
        device=0  # GPU ID
    )
```

> ✅ 训练完成后，最佳模型保存在：`runs/train/fire_yolov5s/weights/best.pt`

---

## 🎥 四、火焰检测推理系统（支持图片/视频/摄像头）

将以下代码保存为 `fire_detection.py`：

```python
# fire_detection.py
import cv2
import torch
import numpy as np
import argparse
import time
import os
from pathlib import Path

# 报警函数（可替换为短信/邮件/声光报警）
def trigger_alarm(frame, save_dir="alarms"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"{save_dir}/alarm_{timestamp}.jpg", frame)
    print(f"🔥 火灾警报！已保存截图: {save_dir}/alarm_{timestamp}.jpg")
    # 可添加：播放警报音、发送 HTTP 请求等

def detect(source, weights, conf_thres=0.5, alarm_threshold=1):
    """
    :param source: 图片路径 / 视频路径 / '0'（摄像头）
    :param weights: 模型路径（.pt）
    :param conf_thres: 置信度阈值
    :param alarm_threshold: 触发报警的最小火焰数量
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    model.conf = conf_thres

    # 判断输入类型
    is_camera = source == '0'
    is_video = source.endswith(('.mp4', '.avi'))
    
    if is_camera or is_video:
        cap = cv2.VideoCapture(0 if is_camera else source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1.0 / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()

            fire_count = 0
            for *xyxy, conf, cls in detections:
                if int(cls) == 0:  # class 0 = fire
                    fire_count += 1
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Fire {conf:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 报警逻辑
            if fire_count >= alarm_threshold:
                trigger_alarm(frame)

            # 显示状态
            status = "🔥 DANGER!" if fire_count >= alarm_threshold else "✅ SAFE"
            cv2.putText(frame, f"Status: {status} | Fire: {fire_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 控制帧率
            elapsed = time.time() - start_time
            delay = max(1, int((frame_time - elapsed) * 1000))
            cv2.imshow("Fire Detection", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        # 单张图片检测
        img = cv2.imread(source)
        results = model(img)
        results.render()
        cv2.imshow("Result", results.ims[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='image/video/camera')
    parser.add_argument('--weights', type=str, default='best.pt', help='model path')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    args = parser.parse_args()

    detect(args.source, args.weights, args.conf)
```

### 运行示例：

```bash
# 摄像头实时检测
python fire_detection.py --source 0 --weights runs/train/fire_yolov5s/weights/best.pt

# 视频检测
python fire_detection.py --source test.mp4 --weights best.pt

# 图片检测
python fire_detection.py --source test.jpg --weights best.pt
```

---

## 🚀 五、部署到 NVIDIA Jetson Nano/NX

### 步骤 1：在 Jetson 上安装依赖

```bash
# 安装 JetPack（含 CUDA/TensorRT）
# 安装 OpenCV（通常已预装）
sudo apt install python3-opencv

# 安装 PyTorch（Jetson 版本）
wget https://nvidia.box.com/shared/static/...  # 从 NVIDIA 官网下载 wheel
pip3 install torch-*.whl
```

### 步骤 2：导出为 TensorRT（加速推理）

```python
# export_to_trt.py
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
import torch

device = select_device('0')
model = attempt_load('best.pt', map_location=device)
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 3, 640, 640).to(device)
torch.onnx.export(model, dummy_input, "fire.onnx", opset_version=12)

# 然后使用 trtexec 转 TensorRT（Jetson 上执行）：
# /usr/src/tensorrt/bin/trtexec --onnx=fire.onnx --saveEngine=fire.engine
```

### 步骤 3：Jetson 上运行轻量推理（使用 TensorRT）

> 推荐使用 [TensorRT-YOLOv5](https://github.com/wang-xinyu/tensorrtx) 项目加载 `.engine` 文件，实现 **>20 FPS @ Jetson Nano**。

---

## ⚙️ 六、性能优化建议

| 场景 | 建议 |
|------|------|
| **高帧率** | 使用 `yolov5n` 或 `yolov5s` + `imgsz=416` |
| **高精度** | 使用 `yolov5m` + `imgsz=640` + TTA（测试时增强） |
| **夜间火焰** | 在训练集中加入红外/低光照样本 |
| **减少误报** | 增加 `smoke` 类别，结合火焰+烟雾联合判断 |

---


