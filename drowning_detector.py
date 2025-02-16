import cv2
import torch
import numpy as np
from collections import deque
from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class DrowningDetector:
    """
    封装检测、跟踪、溺水分类等逻辑的类
    """

    def __init__(self,
                 model_path="model/best_drown_model_EfficientNet_B3.pth",
                 yolo_weights="yolov8l.pt",
                 skip_frames=10,
                 conf_thres=0.3,
                 iou_thres=0.4,
                 alert_threshold=0.95,
                 history_length=6):
        """
        :param model_path:    溺水分类模型路径
        :param yolo_weights:  YOLOv8 模型权重
        :param skip_frames:   每多少帧做一次检测
        :param conf_thres:    YOLO置信度阈值
        :param iou_thres:     YOLO iou 阈值
        :param alert_threshold: 判定溺水的概率阈值
        :param history_length: 每个目标保留多少帧的历史溺水概率
        """

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DrowningDetector] Using device: {self.DEVICE}")

        # 1) 加载溺水分类模型 (EfficientNet-B3)
        self.drown_model = self._load_drowning_model(model_path)

        # 2) 加载 YOLOv8 & DeepSORT
        self.yolo_model = YOLO(yolo_weights)
        self.tracker = DeepSort(max_age=10, n_init=5,
                                nn_budget=50, max_iou_distance=0.9)

        # 3) 初始化参数
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.alert_threshold = alert_threshold
        self.history_length = history_length

        # 4) 历史记录： {track_id: deque([...])}
        self.history = {}

        # 5) 预处理 transform
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 定义告警级别可自己调整
        self.YELLOW_ALERT_COUNT = 1
        self.RED_ALERT_COUNT = 3
        self.CRITICAL_ALERT_COUNT = 5

    def _load_drowning_model(self, model_path):
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)  # 2分类: [normal, drown]
        model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        model.to(self.DEVICE)
        model.eval()
        print(f"[DrowningDetector] Loaded drown_model from {model_path}")
        return model

    def process_frame(self, frame):
        """
        对单帧执行：
        1) 判断是否到达 skip_frames（若没到，直接返回原帧）
        2) YOLO 检测 + DeepSORT 跟踪
        3) 溺水分类
        4) 绘制边框/提示
        5) 返回处理后的帧
        """
        self.frame_count += 1
        # 如果没到检测帧，直接返回原帧
        if self.frame_count % self.skip_frames != 0:
            return frame

        # ---------- YOLO 检测 ----------
        results = self.yolo_model(frame, conf=self.conf_thres, iou=self.iou_thres)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # 仅处理 person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], conf, "person"))

        # ---------- DeepSORT 跟踪 ----------
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # ---------- 对每个目标做溺水分类 ----------
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

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # 溺水分类
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(roi_rgb)
            img = self.transform(img).unsqueeze(0).to(self.DEVICE)
            with torch.no_grad():
                output = self.drown_model(img)
                probs = torch.softmax(output, dim=1)
                drown_prob = probs[0, 1].item()

            # 维护历史
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.history_length)
            self.history[track_id].append(drown_prob)

            # 计算报警等级
            alert_count = sum(p > self.alert_threshold for p in self.history[track_id])
            critical_alert = (alert_count >= self.CRITICAL_ALERT_COUNT)
            red_alert = (alert_count >= self.RED_ALERT_COUNT)
            yellow_alert = (alert_count >= self.YELLOW_ALERT_COUNT)

            # 设置框颜色
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
            cv2.putText(frame, f"Drown: {drown_prob:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 报警文字
            if red_alert:
                cv2.putText(frame, "ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if critical_alert:
                cv2.putText(frame, "CRITICAL ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        return frame
