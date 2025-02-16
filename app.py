import os
import cv2
import torch
import threading
from time import sleep
from collections import deque
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for, send_file, Response, jsonify
from flask_cors import CORS

# YOLO & DeepSORT
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# TorchVision: EfficientNet-B3
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# =================== 1. 定义溺水检测器 ===================
class DrowningDetector:
    def __init__(self,
                 model_path="model/best_drown_model_EfficientNet_B3.pth",
                 yolo_weights="yolov8l.pt",
                 skip_frames=10,
                 padding=20,
                 conf_thres=0.3,
                 iou_thres=0.4,
                 alert_threshold=0.95,
                 history_length=6):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[DrowningDetector] Using device:", self.DEVICE)

        # 1) EfficientNet-B3 溺水分类模型
        self.drown_model = self._load_drown_model(model_path)

        # 2) YOLOv8 + DeepSORT
        self.yolo_model = YOLO(yolo_weights)
        self.tracker = DeepSort(max_age=10, n_init=5, nn_budget=50, max_iou_distance=0.9)

        # 3) 主要参数
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.padding = padding
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.alert_threshold = alert_threshold
        self.history_length = history_length

        # 历史记录: {track_id: deque([prob, prob, ...])}
        self.history = {}

        # 用于“非检测帧”也显示上次结果
        self.last_processed_frame = None

        # 告警等级计数
        self.YELLOW_ALERT_COUNT = 1
        self.RED_ALERT_COUNT = 3
        self.CRITICAL_ALERT_COUNT = 5

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_drown_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Drowning model not found: {model_path}")
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)  # 二分类
        model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        model.to(self.DEVICE)
        model.eval()
        print("[DrowningDetector] Loaded drown model:", model_path)
        return model

    def process_frame(self, frame):
        """跳帧逻辑：到达 skip_frames 或首次，就检测；否则复用上次结果。"""
        self.frame_count += 1
        # 每 skip_frames 帧做一次检测，或者首次没处理过
        if (self.frame_count % self.skip_frames == 1) or (self.last_processed_frame is None):
            processed = self._detect_and_draw(frame)
            self.last_processed_frame = processed
            return processed
        else:
            # 非检测帧: 直接返回上次绘制好的结果
            return self.last_processed_frame

    def _detect_and_draw(self, frame):
        """对单帧执行 YOLO+DeepSORT+溺水分类 并 画框。"""
        h, w, _ = frame.shape
        results = self.yolo_model(frame, conf=self.conf_thres, iou=self.iou_thres)
        detections = []

        # 1) YOLO 检测到 person，加 padding
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, x1 - self.padding)
                    y1 = max(0, y1 - self.padding)
                    x2 = min(w, x2 + self.padding)
                    y2 = min(h, y2 + self.padding)
                    conf = float(box.conf[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        # 2) DeepSORT 跟踪
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # 3) 对每个目标做溺水分类
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w-1, x2); y2 = min(h-1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # 溺水分类
            drown_prob = self._classify_drowning(roi)

            # 更新历史
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.history_length)
            self.history[track_id].append(drown_prob)

            # 告警等级
            alert_count = sum(p > self.alert_threshold for p in self.history[track_id])
            critical_alert = (alert_count >= self.CRITICAL_ALERT_COUNT)
            red_alert = (alert_count >= self.RED_ALERT_COUNT)
            yellow_alert = (alert_count >= self.YELLOW_ALERT_COUNT)

            # 颜色
            if critical_alert:
                color = (0, 0, 255)
            elif red_alert:
                color = (0, 0, 200)
            elif yellow_alert:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            # 绘制
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Drown: {drown_prob:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if red_alert:
                cv2.putText(frame, "ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
            if critical_alert:
                cv2.putText(frame, "CRITICAL ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 4)

        return frame

    def _classify_drowning(self, roi_bgr):
        """用 EfficientNet-B3 对ROI做溺水分类，返回溺水概率"""
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            output = self.drown_model(tensor)
            probs = torch.softmax(output, dim=1)
            drown_prob = probs[0, 1].item()
        return drown_prob


# =================== 2. Flask 应用 ===================
app = Flask(__name__, template_folder="templates")
CORS(app)

# 全局进度状态，用于轮询
progress_status = {"progress": 0, "done": False, "filename": ""}

# 全局“实时检测”用的检测器
global_detector = DrowningDetector(
    model_path="model/best_drown_model_EfficientNet_B3.pth",
    yolo_weights="yolov8l.pt",
    skip_frames=10,
    padding=20,
    conf_thres=0.3,
    iou_thres=0.4,
    alert_threshold=0.95,
    history_length=6
)


@app.route("/")
def index():
    """ 首页 """
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    """ 上传视频 """
    if request.method == "GET":
        return render_template("upload.html")

    if "video" not in request.files:
        return "No video file!", 400

    file = request.files["video"]
    filename = file.filename
    if not filename:
        return "No selected file!", 400

    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", filename)
    file.save(save_path)
    print("[upload_video] Saved to:", save_path)

    # 跳转到 result 页面
    return redirect(url_for("result_page", file=filename))


@app.route("/result")
def result_page():
    """
    显示上传后的视频名，并提供：
    1) 实时播放 (IMG MJPEG)
    2) 离线处理 (带进度条)，完成后可下载
    """
    filename = request.args.get("file")
    if not filename:
        return "Missing file parameter!", 400
    return render_template("result.html", filename=filename)


@app.route("/processing_stream")
def processing_stream():
    """
    实时播放：读取上传文件 -> 边检测边推流
    """
    filename = request.args.get("file")
    if not filename:
        return "Missing file param!", 400

    input_path = os.path.join("uploads", filename)
    if not os.path.exists(input_path):
        return f"File not found: {filename}", 404

    def gen_frames(path):
        cap = cv2.VideoCapture(path)
        # 重置 global_detector 的计数和历史
        global_detector.frame_count = 0
        global_detector.history.clear()
        global_detector.last_processed_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = global_detector.process_frame(frame)

            ret2, buffer = cv2.imencode(".jpg", processed)
            if not ret2:
                continue
            yield (b"--frame\r\n" +
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

        cap.release()

    return Response(gen_frames(input_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- 进度条相关路由 ----------------
@app.route("/download_processed")
def download_processed():
    """
    异步离线处理：更新进度进度条 -> 处理完成后可下载
    这里不直接send_file，而是用progress_status表示状态
    前端JS每隔1秒轮询 /process_status
    """
    global progress_status
    filename = request.args.get("file")
    if not filename:
        return jsonify({"error": "Missing file parameter"}), 400

    input_path = os.path.join("uploads", filename)
    if not os.path.exists(input_path):
        return jsonify({"error": "File not found"}), 404

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"processed_{filename}")

    # 重置进度
    progress_status = {"progress": 0, "done": False, "filename": filename}

    # 开线程处理，不阻塞 Flask
    threading.Thread(target=process_video_offline, args=(input_path, output_path)).start()

    return jsonify({"status": "processing", "filename": filename})


@app.route("/process_status")
def process_status_api():
    """ 前端轮询进度条时，会请求这里 """
    global progress_status
    return jsonify(progress_status)


def process_video_offline(input_path, output_path):
    """
    在后台线程中，离线处理视频 => 写出 processed_{filename}.mp4
    """
    global progress_status

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 用一个本地检测器, 与全局分开
    local_detector = DrowningDetector(
        model_path="model/best_drown_model_EfficientNet_B3.pth",
        yolo_weights="yolov8l.pt",
        skip_frames=10,
        padding=20,
        conf_thres=0.1,
        iou_thres=0.4,
        alert_threshold=0.95,
        history_length=6
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = local_detector.process_frame(frame)
        out.write(processed)

        processed_frames += 1
        progress_status["progress"] = int(processed_frames / total_frames * 100)

    cap.release()
    out.release()

    progress_status["progress"] = 100
    progress_status["done"] = True
    print("[process_video_offline] Done ->", output_path)


# =================== 启动 ===================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
