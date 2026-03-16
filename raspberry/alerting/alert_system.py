#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alert System \u2014 Raspberry Pi Edge IDS.

Multi-channel notification service triggered when the IDS detects attacks.
Supports Telegram Bot API, SMTP email, and Slack Incoming Webhooks.

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO, ALERT_EMAIL_FROM,
    WEBHOOK_URL,
)


class AlertSystem:
    """
    Multi-channel alert system for IDS attack notifications.
    Sends alerts via Telegram, Email, and/or Webhook based on configuration.
    """

    def __init__(self):
        self.channels = []

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            self.channels.append("telegram")
        if SMTP_USER and SMTP_PASSWORD and ALERT_EMAIL_TO:
            self.channels.append("email")
        if WEBHOOK_URL:
            self.channels.append("webhook")

        if self.channels:
            print(f"[OK] Alert System initialized: {', '.join(self.channels)}")
        else:
            print("[WARN] Alert System: No channels configured (check .env)")

    def send_all(self, message: str):
        """Send alert through all configured channels."""
        results = {}
        for channel in self.channels:
            try:
                if channel == "telegram":
                    self.send_telegram(message)
                    results["telegram"] = "[OK]"
                elif channel == "email":
                    self.send_email("[ALERT] IDS Alert - Attack Detected", message)
                    results["email"] = "[OK]"
                elif channel == "webhook":
                    self.send_webhook(message)
                    results["webhook"] = "[OK]"
            except Exception as e:
                results[channel] = f"[ERR] {e}"
                print(f"  [WARN] Alert [{channel}] failed: {e}")

        return results

    # ------------------------------------------------------------------
    # TELEGRAM
    # ------------------------------------------------------------------
    def send_telegram(self, message: str):
        """Send a message via Telegram Bot API."""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"  [OK] Telegram alert sent")

    # ------------------------------------------------------------------
    # EMAIL (SMTP)
    # ------------------------------------------------------------------
    def send_email(self, subject: str, body: str):
        """Send an email alert via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = ALERT_EMAIL_FROM or SMTP_USER
        msg["To"] = ALERT_EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"  [OK] Email alert sent to {ALERT_EMAIL_TO}")

    # ------------------------------------------------------------------
    # WEBHOOK (Slack)
    # ------------------------------------------------------------------
    def send_webhook(self, message: str):
        """Send alert via Slack Incoming Webhook."""
        payload = {
            "text": f":rotating_light: *IDS ALERT - ATTACK DETECTED*\n```{message}```",
            "username": "IDS Edge - Raspberry Pi",
            "icon_emoji": ":shield:",
        }
        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        print(f"  [OK] Slack webhook alert sent")
