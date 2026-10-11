import streamlit as st
import json
import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="Shelf Sight, Stock Right", layout="wide")
st.title("🛒 Shelf Sight, Stock Right")
st.caption("Real-time shelf audit dashboard")

# ==========================================
# LOAD PLANOGRAM
# ==========================================
planogram = json.load(open("planogram.json"))

with st.sidebar:
    st.header("Planogram")
    st.json(planogram, expanded=False)

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if "initial_stock" not in st.session_state:
    st.session_state.initial_stock = {}
if "slot_counts" not in st.session_state:
    st.session_state.slot_counts = {}
if "slot_labels" not in st.session_state:
    st.session_state.slot_labels = {}
if "annotated_img" not in st.session_state:
    st.session_state.annotated_img = None

# ==========================================
# SECTION 1: MANUAL STOCK INPUT (KEYBOARD)
# ==========================================
st.subheader("1️⃣ Enter Initial Stock (Manual Input)")
st.write("Shelf par kitne items hone chahiye, woh yahan enter karein:")

with st.form("stock_input_form"):
    for shelf in planogram["shelves"]:
        for slot in shelf["slots"]:
            sid = slot["slot_id"]
            expected = slot["sku"]
            # Default value 0 agar pehle se save nahi hai
            default_val = st.session_state.initial_stock.get(sid, 0)
            val = st.number_input(f"{sid} ({expected})", min_value=0, step=1, value=default_val, key=f"input_{sid}")
            st.session_state.initial_stock[sid] = val
    submitted = st.form_submit_button("Save Initial Stock")

if submitted:
    st.success("✅ Initial stock saved! Now take a photo below to check.")

# ==========================================
# SECTION 2: CAMERA DETECTION
# ==========================================
st.write("---")
st.subheader("2️⃣ Capture Shelf Image (Camera)")
st.write("Click 'Take Photo' to capture a shelf image and run real-time detection.")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

img_file_buffer = st.camera_input("Take a picture of the shelf")

if img_file_buffer is not None:
    # Convert to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Run YOLO detection
    with st.spinner("Detecting products..."):
        results = model.predict(cv2_img, conf=0.35)
        st.session_state.annotated_img = results[0].plot()
    
    # Map to planogram slots
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detections.append({
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "label": label
        })

    # Reset counts for new image
    st.session_state.slot_counts = {}
    st.session_state.slot_labels = {}
    h, w = cv2_img.shape[:2]
    
    for shelf in planogram["shelves"]:
        for slot in shelf["slots"]:
            sid = slot["slot_id"]
            st.session_state.slot_counts[sid] = 0
            st.session_state.slot_labels[sid] = []
            
            sx1 = int(slot["x"] * w)
            sx2 = int((slot["x"] + slot["w"]) * w)
            sy1 = int(shelf["y"] * h)
            sy2 = int((shelf["y"] + shelf["height"]) * h)
            
            for det in detections:
                cx, cy = det["center"]
                if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                    st.session_state.slot_counts[sid] += 1
                    st.session_state.slot_labels[sid].append(det["label"])

# ==========================================
# SECTION 3: CALCULATION & METRICS
# ==========================================
if st.session_state.slot_counts:
    st.write("---")
    st.subheader("3️⃣ Audit Results")
    
    total_missing = 0
    total_initial = 0
    total_detected = 0
    
    for shelf in planogram["shelves"]:
        for slot in shelf["slots"]:
            sid = slot["slot_id"]
            initial = st.session_state.initial_stock.get(sid, 0)
            detected = st.session_state.slot_counts.get(sid, 0)
            
            total_initial += initial
            total_detected += detected
            
            # Missing = Initial - Detected (Agar negative ho toh 0)
            missing = max(0, initial - detected)
            total_missing += missing

    c1, c2, c3 = st.columns(3)
    c1.metric("📦 Total Initial Stock", total_initial)
    c2.metric("📷 Total Detected", total_detected)
    c3.metric("🚨 Total Missing/Sold", total_missing, delta=f"-{total_missing}" if total_missing > 0 else "0")

    if st.session_state.annotated_img is not None:
        st.image(st.session_state.annotated_img, channels="BGR", caption="Detected Products")

    st.write("---")
    st.subheader("📊 Detailed Slot Status")
    
    for shelf in planogram["shelves"]:
        for slot in shelf["slots"]:
            sid = slot["slot_id"]
            expected = slot["sku"]
            initial = st.session_state.initial_stock.get(sid, 0)
            detected = st.session_state.slot_counts.get(sid, 0)
            missing = max(0, initial - detected)
            detected_labels = st.session_state.slot_labels.get(sid, [])
            
            if detected == 0 and initial > 0:
                st.error(f"🔴 **{sid}** ({expected}) → **Out of Stock** | Initial: {initial} | Detected: 0 | Missing: {missing}")
            elif missing > 0:
                st.warning(f"🟠 **{sid}** ({expected}) → **Missing {missing}** | Initial: {initial} | Detected: {detected}")
            elif any(l != expected and l != "object" for l in detected_labels):
                st.warning(f"🟡 **{sid}** ({expected}) → **Misplaced** (Detected: {detected_labels})")
            else:
                st.success(f"🟢 **{sid}** ({expected}) → **OK** | Initial: {initial} | Detected: {detected}")
