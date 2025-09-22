#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def send_webhook(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        resp.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send JSON payload to webhook if URL available")
    parser.add_argument("--message", type=str, required=True)
    parser.add_argument("--url", type=str, default=os.environ.get("ALERT_WEBHOOK_URL"))
    args = parser.parse_args()

    if not args.url:
        print("[notify] No webhook URL provided; skipping", file=sys.stderr)
        return

    payload = {"text": args.message}
    try:
        send_webhook(args.url, payload)
        print("[notify] sent")
    except (URLError, HTTPError) as exc:
        print(f"[notify] failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
