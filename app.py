import os
import cv2
import torch
import numpy as np
from collections import deque
from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from flask import Flask, render_template, Response

# =========== Flask 初始化 ===========
app = Flask(__name__, template_folder="templates")

# =========== 设备选择 ===========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========== 1. 加载溺水检测模型 (EfficientNet-B3) ===========
def load_drowning_model(model_path="model/best_drown_model_EfficientNet_B3.pth"):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    # 替换最后一层，做二分类 (0=正常, 1=溺水)
    model.classifier[1] = torch.nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

drown_model = load_drowning_model("model/best_drown_model_EfficientNet_B3.pth")

# =========== 2. 加载 YOLOv8 + DeepSORT ===========
yolo_model = YOLO("yolov8l.pt")  # 第一次运行会自动下载
tracker = DeepSort(max_age=10, n_init=5, nn_budget=50, max_iou_distance=0.9)

# 预处理：对裁剪的人体区域做溺水分类
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========== 3. 历史记录 & 阈值 ===========
history = {}
HISTORY_LENGTH = 6        # 记录最近6次溺水概率
ALERT_THRESHOLD = 0.95
YELLOW_ALERT_COUNT = 1
RED_ALERT_COUNT = 3
CRITICAL_ALERT_COUNT = 5

# =========== 4. 实时检测的生成器函数 ===========
def gen_frames():
    """
    不断读取摄像头或视频文件，做YOLO检测 & DeepSORT跟踪 & 溺水识别，
    并把处理后的帧以流的形式发送给前端。
    """
    # 如果用摄像头：
    cap = cv2.VideoCapture(0)  # 或者 cap = cv2.VideoCapture("你的视频路径.mp4")

    frame_count = 0
    skip_frames = 5  # 可以自定义：每隔多少帧做一次检测

    while True:
        success, frame = cap.read()
        if not success:
            # 视频播放完毕或摄像头异常时，跳出循环
            break

        frame_count += 1

        # 如果帧数未到达 skip_frames，直接原帧返回
        # （这样可以减少计算量，但会降低检测频率）
        if frame_count % skip_frames != 0:
            # 把当前帧编码成JPEG再yield
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        # =========== YOLOv8 检测 ===========
        results = yolo_model(frame, conf=0.3, iou=0.4)
        detections = []

        # 从YOLO结果中抽取 person
        for r in results:
            for box in r.boxes:
                if int(box.cls) == 0:  # 仅处理类别 "person"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    w = x2 - x1
                    h = y2 - y1
                    # 保存到DeepSORT要求的格式: [x, y, w, h], conf, class
                    detections.append(([x1, y1, w, h], conf, "person"))

        # =========== DeepSORT 跟踪 ===========
        tracks = tracker.update_tracks(detections, frame=frame)

        # =========== 对每个Track做溺水分类 ===========
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # 防止越界
            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 取出ROI
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            # 前处理 + 溺水分类
            img = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = drown_model(img)
                probs = torch.softmax(output, dim=1)
                drown_prob = probs[0, 1].item()

            # 维护历史
            if track_id not in history:
                history[track_id] = deque(maxlen=HISTORY_LENGTH)
            history[track_id].append(drown_prob)

            # 根据溺水概率次数判断警报等级
            alert_count = sum(p > ALERT_THRESHOLD for p in history[track_id])
            yellow_alert = (alert_count >= YELLOW_ALERT_COUNT)
            red_alert = (alert_count >= RED_ALERT_COUNT)
            critical_alert = (alert_count >= CRITICAL_ALERT_COUNT)

            # 根据警报等级设置框颜色
            if critical_alert:
                color = (0, 0, 255)     # **红色（最高危）**
            elif red_alert:
                color = (0, 0, 200)    # **深红色（高危）**
            elif yellow_alert:
                color = (0, 255, 255)  # **黄色（预警）**
            else:
                color = (0, 255, 0)    # **绿色（正常）**

            # 绘制跟踪框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Drown: {drown_prob:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 如果达到警报等级，顶部显示文字
            if red_alert:
                cv2.putText(frame, "ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if critical_alert:
                cv2.putText(frame, "CRITICAL ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # 把当前处理后的帧编码成JPEG再yield
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# =========== 5. 路由：主页 & 视频流接口 ===========
@app.route('/')
def index():
    """ 返回主页HTML，内含 <img> 标签展示 /video_feed """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    把 gen_frames() 生成的图像流，通过 multipart/x-mixed-replace 推送给浏览器
    浏览器前端用 <img src="/video_feed"> 即可显示实时画面
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =========== 6. 启动 Flask ===========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
