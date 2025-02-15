import os
import cv2
import torch
import numpy as np
from collections import deque
from PIL import Image
from flask import Flask, render_template, request, Response, jsonify, send_file
from flask_cors import CORS

# YOLOv8 & DeepSORT
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# TorchVision: EfficientNet-B3
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ------------------ 初始化 Flask ------------------
app = Flask(__name__, template_folder="templates")
CORS(app)

# ------------------ 设备选择 ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ------------------ 1. 加载溺水检测模型 (EfficientNet-B3) ------------------
def load_drowning_model(model_path="model/best_drown_model_EfficientNet_B3.pth"):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    # 修改分类器头，做二分类 (0=正常, 1=溺水)
    model.classifier[1] = torch.nn.Linear(in_features, 2)
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

drown_model = load_drowning_model("model/best_drown_model_EfficientNet_B3.pth")

# ------------------ 2. 加载 YOLOv8 + DeepSORT 跟踪器 ------------------
yolo_model = YOLO("yolov8l.pt")  # 第一次运行会自动下载 yolov8l.pt
tracker = DeepSort(max_age=10, n_init=5, nn_budget=50, max_iou_distance=0.9)

# ------------------ 3. 图像预处理 & 历史记录阈值 ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

history = {}               # {track_id: deque([...])}
HISTORY_LENGTH = 6         # 记录最近6次的溺水概率
ALERT_THRESHOLD = 0.95
YELLOW_ALERT_COUNT = 1
RED_ALERT_COUNT = 3
CRITICAL_ALERT_COUNT = 5

# ==================== 路由 1：主页面 ====================
@app.route("/")
def index():
    """
    返回首页，可简单放一些链接，比如“实时检测”与“上传视频”。
    """
    return render_template("index.html")

# ==================== 路由 2：实时检测页面 ====================
@app.route("/realtime")
def realtime():
    """
    返回一个页面，其中用 <img src="/video_feed" /> 来显示实时视频流
    """
    return render_template("realtime.html")

# ==================== 路由 2.1：实时视频流接口 ====================
def gen_frames():
    """
    不断读取摄像头/视频文件，对每帧执行 YOLO 检测 + DeepSORT 跟踪 + 溺水分类，
    最终将处理后的帧编码成 JPEG，通过 yield 以 MJPEG 流的形式发送到前端。
    """
    # 你可以改为从本地文件读取，比如：cap = cv2.VideoCapture("LifeguardRescueVideos/video_72.mp4")
    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

    frame_count = 0
    skip_frames = 5  # 每隔多少帧做一次检测，减少计算量

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            # 不到检测时机，直接送原帧
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        # ------------------ YOLOv8 检测 ------------------
        results = yolo_model(frame, conf=0.3, iou=0.4)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 0:  # 仅检测 person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], conf, "person"))

        # ------------------ DeepSORT 跟踪 ------------------
        tracks = tracker.update_tracks(detections, frame=frame)

        # ------------------ 对每个跟踪目标做溺水判断 ------------------
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            H, W, _ = frame.shape
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            # 溺水分类：EfficientNet-B3
            img = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = drown_model(img)
                probs = torch.softmax(output, dim=1)
                drown_prob = probs[0, 1].item()

            # 更新历史
            if track_id not in history:
                history[track_id] = deque(maxlen=HISTORY_LENGTH)
            history[track_id].append(drown_prob)

            # 计算报警等级
            alert_count = sum(p > ALERT_THRESHOLD for p in history[track_id])
            critical_alert = (alert_count >= CRITICAL_ALERT_COUNT)
            red_alert = (alert_count >= RED_ALERT_COUNT)
            yellow_alert = (alert_count >= YELLOW_ALERT_COUNT)

            # 决定框颜色
            if critical_alert:
                color = (0, 0, 255)     # 红色
            elif red_alert:
                color = (0, 0, 200)    # 深红
            elif yellow_alert:
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 255, 0)    # 绿色

            # 绘制
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Drown: {drown_prob:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 提示文字
            if red_alert:
                cv2.putText(frame, "ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if critical_alert:
                cv2.putText(frame, "CRITICAL ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # 把处理后的帧传给浏览器
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
    """
    实时视频流接口，返回 gen_frames() 生成的 MJPEG 流
    前端只需 <img src="/video_feed" /> 即可实时显示
    """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================== 路由 3：上传视频 ====================
@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    """
    GET 时返回上传页面，POST 时接收视频、处理并返回处理后的文件
    """
    if request.method == "GET":
        # 渲染一个简单的上传页面
        return render_template("upload.html")

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    filename = file.filename

    # 1) 保存原视频
    os.makedirs("uploads", exist_ok=True)
    input_path = os.path.join("uploads", filename)
    file.save(input_path)

    # 2) 处理视频
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"processed_{filename}")
    process_video(input_path, output_path)

    # 3) 返回处理后的视频
    return send_file(output_path, as_attachment=True,
                     download_name=f"processed_{filename}")

def process_video(input_path, output_path):
    """
    对上传的视频做 YOLO 检测 + DeepSORT 跟踪 + 溺水识别，
    写到新的视频文件 output_path
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    skip_frames = 10

    # 对于“上传处理”，可以使用独立的 history dict，
    # 以免与实时检测时的历史混淆:
    local_history = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % skip_frames != 0:
            out.write(frame)
            continue

        # YOLO 检测
        results = yolo_model(frame, conf=0.3, iou=0.4)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    w_box = x2 - x1
                    h_box = y2 - y1
                    detections.append(([x1, y1, w_box, h_box], conf, "person"))

        # DeepSORT
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # 防越界
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(x2, w-1); y2 = min(y2, h-1)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # 溺水分类
            img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = drown_model(img)
                probs = torch.softmax(output, dim=1)
                drown_prob = probs[0, 1].item()

            # 本地 history
            if track_id not in local_history:
                local_history[track_id] = deque(maxlen=HISTORY_LENGTH)
            local_history[track_id].append(drown_prob)

            # 计算报警等级
            alert_count = sum(p > ALERT_THRESHOLD for p in local_history[track_id])
            critical_alert = (alert_count >= CRITICAL_ALERT_COUNT)
            red_alert = (alert_count >= RED_ALERT_COUNT)
            yellow_alert = (alert_count >= YELLOW_ALERT_COUNT)

            # 颜色
            if critical_alert:
                color = (0, 0, 255)
            elif red_alert:
                color = (0, 0, 200)
            elif yellow_alert:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Drown: {drown_prob:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if red_alert:
                cv2.putText(frame, "ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if critical_alert:
                cv2.putText(frame, "CRITICAL ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        out.write(frame)

    cap.release()
    out.release()

# ==================== 主函数 ====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
