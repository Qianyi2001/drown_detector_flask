import os
import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from collections import deque
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 初始化 Flask
app = Flask(__name__)
CORS(app)

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入 EfficientNet-B3 溺水分类模型
def load_drowning_model(model_path="model/best_drown_model_EfficientNet_B3.pth"):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 2)  # 2分类
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
drown_model = load_drowning_model("model/best_drown_model_EfficientNet_B3.pth")

# 载入 YOLOv8 模型 & DeepSORT 跟踪器
yolo_model = YOLO("yolov8l.pt")
tracker = DeepSort(max_age=10, n_init=5, nn_budget=50, max_iou_distance=0.9)

# 维护溺水概率历史记录
history = {}
HISTORY_LENGTH = 6
ALERT_THRESHOLD = 0.95
RED_ALERT_COUNT = 3

# 处理上传的视频
@app.route("/upload", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    video_path = f"./uploads/{video_file.filename}"
    output_path = f"./outputs/processed_{video_file.filename}"

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    frame_count = 0
    skip_frames = 10  # 每隔 10 帧检测一次

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            out.write(frame)
            continue

        results = yolo_model(frame, conf=0.1, iou=0.4)
        detections = []

        for r in results:
            for box in r.boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            img = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = drown_model(img)
                probs = torch.softmax(output, dim=1)
                drown_prob = probs[0, 1].item()

            if track_id not in history:
                history[track_id] = deque(maxlen=HISTORY_LENGTH)
            history[track_id].append(drown_prob)

            alert_count = sum(p > ALERT_THRESHOLD for p in history[track_id])
            color = (0, 255, 0)
            if alert_count >= RED_ALERT_COUNT:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Drown: {drown_prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    return send_file(output_path, mimetype="video/mp4", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
