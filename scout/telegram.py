"""Telegram sender (plain text, auto-split at the 4096-char limit)."""
from __future__ import annotations

import time

import requests

LIMIT = 4000


def split_text(text: str, limit: int = LIMIT) -> list[str]:
    chunks, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # pathological long line
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def pack(sections: list[str], limit: int = LIMIT) -> list[str]:
    """Merge small sections into as few messages as possible."""
    msgs, cur = [], ""
    for s in sections:
        for part in split_text(s, limit):
            if cur and len(cur) + len(part) + 2 <= limit:
                cur += "\n\n" + part
            else:
                if cur:
                    msgs.append(cur)
                cur = part
    if cur:
        msgs.append(cur)
    return msgs


def send(token: str, chat_id: str, messages: list[str]) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for m in messages:
        for attempt in range(3):
            r = requests.post(url, json={"chat_id": chat_id, "text": m,
                                         "disable_web_page_preview": True}, timeout=30)
            if r.ok:
                break
            if r.status_code == 429:
                time.sleep(r.json().get("parameters", {}).get("retry_after", 5))
                continue
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:200]}")
        time.sleep(1)
