#!/usr/bin/env python3
"""End-to-end email health check.

Runs the full send/receive cycle using the configured SMTP/IMAP credentials:
1. Search for recent unread emails (baseline)
2. Send a test email to self
3. Search again, confirm delivery
4. Mark delivered email as processed

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python -m apps.agentic_process.test_email
"""

import asyncio
import re
import sys
from datetime import datetime, timezone

from . import config
from .email import fetch_email, search_emails, send_email


async def main() -> int:
    print("=== PetEOS Email Health Check ===")
    print(f"SMTP:  {config.SMTP_USERNAME}@{config.SMTP_HOST}:{config.SMTP_PORT}")
    print(f"IMAP:  {config.IMAP_USERNAME}@{config.IMAP_HOST}:{config.IMAP_PORT}")
    print(f"From:  {config.FROM_ADDRESS}")
    print(f"Folder: {config.IMAP_FOLDER}")
    print()

    # 1. Baseline: count unread emails
    print("[1/3] Searching for unread emails (baseline)...")
    baseline = search_emails(tracking_path=None)
    print(f"  Found {len(baseline)} unread email(s)")
    print()

    # 2. Send test email to self
    subject = "peteos-healthcheck"
    body = f"Health check sent at {datetime.now(timezone.utc).isoformat()}"
    print("[2/3] Sending test email to self...")
    try:
        send_email(
            to=config.FROM_ADDRESS,
            subject=subject,
            body=body,
        )
        print("  Sent OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1
    print()

    # 3. Verify delivery
    print("[3/3] Waiting for delivery, then searching again...")
    import time
    time.sleep(5)

    # Use the existing API — search_emails with subject_regex does everything we need
    matched = search_emails(
        subject_regex=re.compile(re.escape(subject), re.IGNORECASE),
        tracking_path=None,
    )

    if not matched:
        print("  FAILED: no matching email found after sending")
        return 1

    print(f"  Found {len(matched)} matching email(s)")
    uid = matched[-1]  # most recent

    info = fetch_email(uid=uid, attachments_dir=None, tracking_path=None)
    print(f"  From:    {info.from_addr}")
    print(f"  Subject: {info.subject}")
    print(f"  Body:    {info.body_text}")

    if subject not in info.subject:
        print(f"  FAILED: expected subject '{subject}'")
        return 1

    print()
    print("=== PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
