import cv2
import json
import time
import uuid
import yaml
from collections import deque, Counter

from detector import YoloDetector, ContourDetector
from alerter import Alerter


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_planogram(path="planogram.json"):
    with open(path) as f:
        return json.load(f)


def rect_to_pixels(slot, shelf, frame_shape):
    h, w = frame_shape[:2]
    x1 = int(slot["x"] * w)
    x2 = int((slot["x"] + slot["w"]) * w)
    y1 = int(shelf["y"] * h)
    y2 = int((shelf["y"] + shelf["height"]) * h)
    return x1, y1, x2, y2


def map_detections_to_slots(detections, planogram, frame_shape):
    slot_counts = {}
    slot_labels = {}

    for shelf in planogram["shelves"]:
        for slot in shelf["slots"]:
            sid = slot["slot_id"]
            slot_counts[sid] = 0
            slot_labels[sid] = []

            x1, y1, x2, y2 = rect_to_pixels(slot, shelf, frame_shape)

            for det in detections:
                cx, cy = det["center"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    slot_counts[sid] += 1
                    slot_labels[sid].append(det["label"])

    return slot_counts, slot_labels


def make_alert(alert_type, slot_id, expected, detected, count, severity):
    return {
        "event_id": str(uuid.uuid4()),
        "type": alert_type,
        "slot_id": slot_id,
        "expected_sku": expected,
        "detected_skus": detected,
        "facing_count": count,
        "severity": severity,
        "timestamp": time.time()
    }


class ShelfAuditor:
    def __init__(self, planogram, config):
        self.planogram = planogram
        self.window = config["audit"]["temporal_window"]
        self.stable = config["audit"]["stable_frames"]
        self.history = {}

    def audit(self, slot_counts, slot_labels):
        alerts = []

        for shelf in self.planogram["shelves"]:
            for slot in shelf["slots"]:
                sid = slot["slot_id"]
                expected = slot["sku"]
                min_facings = slot["min_facings"]

                count = slot_counts.get(sid, 0)
                labels = slot_labels.get(sid, [])

                if sid not in self.history:
                    self.history[sid] = deque(maxlen=self.window)
                self.history[sid].append((count, tuple(sorted(labels))))

                if len(self.history[sid]) >= self.stable:
                    stable_state = Counter(self.history[sid]).most_common(1)[0][0]
                    count = stable_state[0]
                    labels = list(stable_state[1])

                if count == 0:
                    alerts.append(make_alert(
                        "OOS", sid, expected, labels, count, "critical"))

                elif count < min_facings:
                    alerts.append(make_alert(
                        "LOW_STOCK", sid, expected, labels, count, "high"))

                wrong = [l for l in labels if l != expected and l != "object"]
                if wrong:
                    alerts.append(make_alert(
                        "MISPLACED", sid, expected, labels, count, "medium"))

        return alerts


def draw_overlay(frame, planogram, slot_counts, alerts):
    alert_slots = {a["slot_id"]: a["type"] for a in alerts}

    for shelf in planogram["shelves"]:
        for slot in shelf["slots"]:
            sid = slot["slot_id"]
            x1, y1, x2, y2 = rect_to_pixels(slot, shelf, frame.shape)

            if sid in alert_slots:
                atype = alert_slots[sid]
                color = (0, 0, 255) if atype == "OOS" else \
                        (0, 165, 255) if atype == "LOW_STOCK" else \
                        (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{sid} ({slot_counts.get(sid, 0)})",
                        (x1 + 4, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    y = 30
    for a in alerts[:5]:
        text = f"{a['type']} @ {a['slot_id']} ({a['expected_sku']})"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 25

    return frame


def main():
    config = load_config()
    planogram = load_planogram()
    alerter = Alerter(config)
    auditor = ShelfAuditor(planogram, config)

    mode = config["detector"]["mode"]
    if mode == "yolo":
        detector = YoloDetector(
            model_path=config["detector"]["model"],
            conf=config["detector"]["conf_threshold"],
            iou=config["detector"]["iou_threshold"]
        )
    else:
        detector = ContourDetector()

    cap = cv2.VideoCapture(config["camera"]["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Shelf Sight running. Press 'q' to quit.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 3 != 0:
            cv2.imshow("Shelf Sight, Stock Right", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        detections = detector.detect(frame)
        slot_counts, slot_labels = map_detections_to_slots(
            detections, planogram, frame.shape)
        alerts = auditor.audit(slot_counts, slot_labels)

        for a in alerts:
            alerter.emit(a)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.putText(frame, d["label"], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        frame = draw_overlay(frame, planogram, slot_counts, alerts)
        cv2.imshow("Shelf Sight, Stock Right", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()