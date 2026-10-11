import streamlit as st
import json
import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

st.set_page_config(page_title="Shelf Sight, Stock Right", layout="wide")
st.title("🛒 Shelf Sight, Stock Right")
st.caption("Real-time shelf audit dashboard")

LOG_FILE = Path("alerts.log")
planogram = json.load(open("planogram.json"))

with st.sidebar:
    st.header("Planogram")
    st.json(planogram, expanded=False)

alerts = []
if LOG_FILE.exists():
    with open(LOG_FILE) as f:
        for line in f:
            try:
                alerts.append(json.loads(line.split(" ", 2)[-1]))
            except Exception:
                pass

alerts = alerts[-200:]
counts = {"OOS": 0, "LOW_STOCK": 0, "MISPLACED": 0}
for a in alerts:
    counts[a["type"]] = counts.get(a["type"], 0) + 1

c1, c2, c3 = st.columns(3)
c1.metric("🔴 Out of Stock", counts.get("OOS", 0))
c2.metric("🟠 Low Stock", counts.get("LOW_STOCK", 0))
c3.metric("🟡 Misplaced", counts.get("MISPLACED", 0))

st.subheader("Recent Alerts")
if alerts:
    st.dataframe(
        [
            {
                "Time": time.strftime("%H:%M:%S", time.localtime(a["timestamp"])),
                "Type": a["type"],
                "Slot": a["slot_id"],
                "Expected": a["expected_sku"],
                "Detected": ", ".join(a["detected_skus"]),
                "Count": a["facing_count"],
                "Severity": a["severity"],
            }
            for a in reversed(alerts[-50:])
        ],
        use_container_width=True,
    )
else:
    st.info("No alerts yet — system is watching.")

if st.button("🔄 Refresh"):
    st.rerun()
# ==========================================
# LIVE CAMERA DEMO SECTION
# ==========================================
st.markdown("---")
st.subheader("📸 Live Shelf Camera Test")
st.write("Click 'Take Photo' to capture a shelf image and run real-time detection.")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # Auto-downloads the model

model = load_yolo()

# Browser camera input
img_file_buffer = st.camera_input("Take a picture of the shelf")

if img_file_buffer is not None:
    # Convert uploaded image to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Run YOLO detection
    with st.spinner("Detecting products..."):
        results = model.predict(cv2_img, conf=0.35)
        annotated_img = results[0].plot()  # Draws boxes on the image
    
        # Display the annotated image
    st.image(annotated_img, channels="BGR", caption="Detected Products")
    
    # --- NEW: MAP DETECTIONS TO PLANOGRAM SLOTS ---
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detections.append({
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "label": label
        })

    # Map to planogram slots
    slot_counts = {}
    slot_labels = {}
    h, w = cv2_img.shape[:2]
    
    for shelf in planogram["shelves"]:
        for slot in shelf["slots"]:
            sid = slot["slot_id"]
            slot_counts[sid] = 0
            slot_labels[sid] = []
            
            # Calculate slot pixel coordinates
            sx1 = int(slot["x"] * w)
            sx2 = int((slot["x"] + slot["w"]) * w)
            sy1 = int(shelf["y"] * h)
            sy2 = int((shelf["y"] + shelf["height"]) * h)
            
            for det in detections:
                cx, cy = det["center"]
                if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                    slot_counts[sid] += 1
                    slot_labels[sid].append(det["label"])

    # --- NEW: DISPLAY LIVE COUNTS ON DASHBOARD ---
    st.subheader("📊 Live Counts from Captured Image")
    
    col1, col2, col3 = st.columns(3)
    
    # Count total issues for the top metrics
    oos_count = sum(1 for sid, c in slot_counts.items() if c == 0)
    low_count = sum(1 for shelf in planogram["shelves"] for slot in shelf["slots"] if 0 < slot_counts.get(slot["slot_id"], 0) < slot["min_facings"])
    mis_count = sum(1 for shelf in planogram["shelves"] for slot in shelf["slots"] if any(l != slot["sku"] and l != "object" for l in slot_labels.get(slot["slot_id"], [])))
    
    col1.metric("🔴 Out of Stock", oos_count)
    col2.metric("🟠 Low Stock", low_count)
    col3.metric("🟡 Misplaced", mis_count)
    
    # Show detailed slot breakdown
    st.write("---")
    for shelf in planogram["shelves"]:
        for slot in shelf["slots"]:
            sid = slot["slot_id"]
            expected = slot["sku"]
            count = slot_counts.get(sid, 0)
            detected = slot_labels.get(sid, [])
            
            if count == 0:
                st.error(f"🔴 **{sid}** ({expected}) → **Out of Stock**")
            elif count < slot["min_facings"]:
                st.warning(f"🟠 **{sid}** ({expected}) → **Low Stock** (Count: {count})")
            elif any(l != expected and l != "object" for l in detected):
                st.warning(f"🟡 **{sid}** ({expected}) → **Misplaced** (Detected: {detected})")
            else:
                st.success(f"🟢 **{sid}** ({expected}) → **OK** (Count: {count})")
