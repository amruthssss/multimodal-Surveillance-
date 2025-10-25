"""Alert sending utilities: email, SMS, (placeholder) voice call with TTS.
Requires proper credentials via environment or config. Non-blocking via threads.
"""
from __future__ import annotations
import os
import threading
import smtplib
from email.message import EmailMessage
from typing import List
from pathlib import Path

try:
    from twilio.rest import Client  # type: ignore
except Exception:  # pragma: no cover
    Client = None  # type: ignore

try:
    from gtts import gTTS  # type: ignore
except Exception:  # pragma: no cover
    gTTS = None  # type: ignore

from config.config import Config


def _send_email_with_attachments(to_email: str, subject: str, body: str, attachments: List[str]):
    if not Config.MAIL_SERVER or not Config.ALERT_FROM_EMAIL:
        return
    msg = EmailMessage()
    msg['From'] = Config.ALERT_FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content(body)
    for path in attachments:
        if not path or not Path(path).exists():
            continue
        with open(path, 'rb') as f:
            data = f.read()
        msg.add_attachment(data, maintype='application', subtype='octet-stream', filename=os.path.basename(path))
    try:
        s = smtplib.SMTP(Config.MAIL_SERVER, Config.MAIL_PORT)
        if Config.MAIL_USE_TLS:
            s.starttls()
        if Config.MAIL_USERNAME and Config.MAIL_PASSWORD:
            s.login(Config.MAIL_USERNAME, Config.MAIL_PASSWORD)
        s.send_message(msg)
        s.quit()
    except Exception:
        pass


def _send_sms(to_number: str, text: str):
    if not Client or not Config.TWILIO_ACCOUNT_SID:
        return
    try:
        client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
        client.messages.create(body=text, from_=Config.TWILIO_FROM, to=to_number)
    except Exception:
        pass


def _generate_tts_file(message_text: str) -> str | None:
    if not gTTS:
        return None
    try:
        tts = gTTS(message_text)
        out_path = Path(Config.UPLOAD_DIR) / f"tts_{os.getpid()}_{abs(hash(message_text))}.mp3"
        tts.save(out_path)
        return str(out_path)
    except Exception:
        return None


def send_alert(owner_email: str, owner_phone: str, event_label: str, clip_path: str | None, image_path: str | None, confidence: float):
    subject = f"[ALERT] {event_label} detected"
    body = f"Event: {event_label}\nConfidence: {confidence:.2f}\nPlease review dashboard for details."
    attachments = [p for p in [clip_path, image_path] if p]
    threading.Thread(target=_send_email_with_attachments, args=(owner_email, subject, body, attachments), daemon=True).start()
    if owner_phone:
        threading.Thread(target=_send_sms, args=(owner_phone, f"ALERT: {event_label} ({confidence:.2f})"), daemon=True).start()
    # Optionally generate TTS file (not automatically called)
    _generate_tts_file(f"Alert: {event_label} detected with confidence {confidence:.0%}")
