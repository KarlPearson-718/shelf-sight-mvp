import streamlit as st
import json
import time
from pathlib import Path

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