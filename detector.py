import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.35, iou=0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.names = self.model.names

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = self.names[cls_id]
            conf = float(box.conf[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "label": label,
                "confidence": conf
            })
        return detections


class ContourDetector:
    def __init__(self, min_area=800):
        self.min_area = min_area

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 40, 120)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 or h < 20:
                continue
            detections.append({
                "bbox": (x, y, x + w, y + h),
                "center": (x + w / 2, y + h / 2),
                "label": "object",
                "confidence": 1.0
            })
        return detections