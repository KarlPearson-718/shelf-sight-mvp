import json
import time
import logging
import requests
from pathlib import Path


class Alerter:
    def __init__(self, config):
        self.cooldown = config["audit"]["alert_cooldown_sec"]
        self.log_file = Path(config["alert"]["log_file"])
        self.tg = config["alert"]["telegram"]
        self.last_sent = {}

        logging.basicConfig(
            filename=str(self.log_file),
            level=logging.INFO,
            format="%(asctime)s %(message)s"
        )

    def _dedup(self, key):
        now = time.time()
        last = self.last_sent.get(key, 0)
        if now - last < self.cooldown:
            return False
        self.last_sent[key] = now
        return True

    def emit(self, alert):
        key = f"{alert['slot_id']}:{alert['type']}"
        if not self._dedup(key):
            return False

        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(alert["severity"], "⚪")
        print(f"{icon} [{alert['type']}] {alert['slot_id']} "
              f"expected={alert['expected_sku']} "
              f"detected={alert['detected_skus']} "
              f"count={alert['facing_count']}")

        logging.info(json.dumps(alert))

        if self.tg.get("enabled"):
            self._send_telegram(alert)

        return True

    def _send_telegram(self, alert):
        emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(alert["severity"], "⚪")
        text = (
            f"{emoji} *{alert['type']}* alert\n"
            f"Slot: `{alert['slot_id']}`\n"
            f"Expected: `{alert['expected_sku']}`\n"
            f"Detected: `{alert['detected_skus']}`\n"
            f"Facings: `{alert['facing_count']}`"
        )
        url = f"https://api.telegram.org/bot{self.tg['bot_token']}/sendMessage"
        try:
            requests.post(url, json={
                "chat_id": self.tg["chat_id"],
                "text": text,
                "parse_mode": "Markdown"
            }, timeout=5)
        except Exception as e:
            print(f"[telegram error] {e}")