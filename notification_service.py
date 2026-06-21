"""
notification_service.py

Listens on Postgres channel 'cv_alerts' and dispatches formatted alerts to
Telegram. Two notification types, each with its own message format and
debounce window:

  SECURITY    — intrusion detections. Sends the cropped intruder image
                (saved at inference time, see save_intruder_crop in the
                inference notebook) alongside zone, time, movement_type,
                severity. Debounced 5 min per zone — an intrusion is rare
                and high-stakes, repeat alerts within a window are noise,
                not new information.

  OPERATIONS  — crowd density. Sends object_count_in_zone + timestamp for
                the zone the rule was defined on. Includes a staffing
                escalation line when count > 8. Debounced 1 min — crowd
                state changes faster and operations teams want closer to
                real-time counts, just not literally per-frame.

Run as a standalone process (NOT inside the notebook): the main loop blocks
forever waiting on Postgres notifications, which would freeze a notebook
kernel if run as a cell.
"""

import json
import logging
import select
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extensions
import requests

DB_CONFIG = {
    "dbname": "retail_cv_deep_dive",
    "user": "postgres",
    "password": "abc123",
    "host": "localhost",
    "port": 5432,
}
TELEGRAM_BOT_TOKEN = "8717722565:AAHDuI6aO3s8n1ilWBGbWOwgHSrruC9kwrY"   # from @BotFather
TELEGRAM_CHAT_ID = "1410539737"       # numeric chat id (see setup notes)


DEBOUNCE_WINDOWS = {
    "SECURITY": timedelta(minutes=5),
    "OPERATIONS": timedelta(minutes=1),
}

STAFFING_THRESHOLD = 8  # object_count_in_zone above this triggers staffing line

SEVERITY_ICON = {
    "CRITICAL": "🔴",
    "WARNING": "🟠",
    "INFO": "🔵",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("notification_service")


# ---------------------------------------------------------------------------
# Telegram senders
# ---------------------------------------------------------------------------

def send_telegram_message(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Telegram sendMessage failed: {e}")
        return False


def send_telegram_photo(image_path: str, caption: str) -> bool:
    """Sends the cropped intruder image with the alert details as caption.
    Falls back to a text-only message if the file is missing or unreadable —
    a missing crop shouldn't mean the security team gets no alert at all."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
                files={"photo": f},
                timeout=10,
            )
        resp.raise_for_status()
        return True
    except (FileNotFoundError, requests.RequestException) as e:
        log.error(f"Telegram sendPhoto failed ({e}); falling back to text-only alert")
        return send_telegram_message(caption + "\n\n_[image unavailable]_")


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

def format_timestamp(ts_str: str) -> str:
    """Postgres sends timestamp as ISO string in the JSON payload; format for display."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(ts_str)


def format_zone_name(zone_name: str) -> str:
    """zones.zone_name is stored underscore-separated (e.g. Renovation_Area_Restricted).
    Display version replaces underscores with spaces."""
    if not zone_name:
        return "unknown zone"
    return zone_name.replace("_", " ")


def format_security_message(payload: dict) -> str:
    severity = payload.get("severity", "CRITICAL")
    icon = SEVERITY_ICON.get(severity, "🔴")
    zone = format_zone_name(payload.get("zone_name"))
    return (
        f"ATTENTION SECURITY 👮🏻‍♂️\n\n"
        f"Intrusion detected\n"
        f"{icon} *{severity}*\n"
        f"Zone: {zone}\n"
        f"Time: {format_timestamp(payload.get('timestamp'))}\n"
        f"Movement: {payload.get('movement_type', 'unknown')}"
    )


def format_operations_message(payload: dict) -> str:
    severity = payload.get("severity", "WARNING")
    icon = SEVERITY_ICON.get(severity, "🟠")
    zone = format_zone_name(payload.get("zone_name"))
    count = payload.get("object_count_in_zone")
    lines = [
        "ATTENTION OPERATIONS 👩🏻‍🍳",
        "",
        "Customer levels high",
        f"{icon} *{severity}*",
        f"Zone: {zone}",
        f"Current count: {count}",
        f"Time: {format_timestamp(payload.get('timestamp'))}",
    ]
    if count is not None and count > STAFFING_THRESHOLD:
        lines.append("")
        lines.append(f"⚠️ Increase staff to at least 3 — count exceeds {STAFFING_THRESHOLD}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------

class DebounceTracker:
    """Tracks last-fired time per (notification_type, zone_name). Each
    notification type gets its own window from DEBOUNCE_WINDOWS."""

    def __init__(self):
        self._last_fired = {}

    def should_fire(self, notification_type: str, zone_name: str) -> bool:
        window = DEBOUNCE_WINDOWS.get(notification_type, timedelta(minutes=5))
        key = f"{notification_type}:{zone_name}"
        now = datetime.utcnow()
        last = self._last_fired.get(key)
        if last is None or (now - last) > window:
            self._last_fired[key] = now
            return True
        return False


# ---------------------------------------------------------------------------
# Main listener loop
# ---------------------------------------------------------------------------

def handle_payload(payload: dict, debouncer: DebounceTracker):
    notification_type = payload.get("notification_type", "OPERATIONS")
    zone_name = payload.get("zone_name", "unknown")

    if not debouncer.should_fire(notification_type, zone_name):
        log.info(f"Debounced: {notification_type}:{zone_name}")
        return

    if notification_type == "SECURITY":
        caption = format_security_message(payload)
        image_path = payload.get("image_path")
        if image_path:
            sent = send_telegram_photo(image_path, caption)
        else:
            log.warning("SECURITY alert missing image_path; sending text-only")
            sent = send_telegram_message(caption + "\n\n_[no image captured]_")
    elif notification_type == "OPERATIONS":
        text = format_operations_message(payload)
        sent = send_telegram_message(text)
    else:
        log.warning(f"Unknown notification_type: {notification_type}")
        return

    if sent:
        log.info(f"Sent {notification_type} alert for {zone_name}")
    else:
        log.error(f"Failed to send {notification_type} alert for {zone_name}")


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("LISTEN cv_alerts;")
    log.info("Listening on channel 'cv_alerts'...")

    debouncer = DebounceTracker()

    try:
        while True:
            if select.select([conn], [], [], 5) == ([], [], []):
                continue

            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                try:
                    payload = json.loads(notify.payload)
                except json.JSONDecodeError:
                    log.warning(f"Could not parse payload: {notify.payload}")
                    continue
                handle_payload(payload, debouncer)

    except KeyboardInterrupt:
        log.info("Shutting down listener.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()