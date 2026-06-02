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
import time
from datetime import datetime, timezone
from pathlib import Path

from apps.agentic_process import config
from apps.agentic_process.email_client import fetch_email, search_emails, send_email


async def main() -> int:
    print("=== PetEOS Email Health Check ===")
    print(f"SMTP:  {config.SMTP_USERNAME}@{config.SMTP_HOST}:{config.SMTP_PORT}")
    print(f"IMAP:  {config.IMAP_USERNAME}@{config.IMAP_HOST}:{config.IMAP_PORT}")
    print(f"From:  {config.FROM_ADDRESS}")
    print(f"Folder: {config.IMAP_FOLDER}")
    print()

    # 1. Baseline: count unread emails
    print("[1/3] Searching for unread emails (baseline)...")
    baseline = search_emails()
    print(f"  Found {len(baseline)} unread email(s)")
    print()

    # 2. Send test email to self with a small attachment
    subject = "peteos-healthcheck"
    body = f"Health check sent at {datetime.now(timezone.utc).isoformat()}"
    print("[2/4] Sending test email to self with attachment...")
    try:
        tmp_path = Path("/tmp/peteos-healthcheck-attachment.txt")
        tmp_path.write_text("Hello from PetEOS — health check attachment")
        send_email(
            to=config.FROM_ADDRESS,
            subject=subject,
            body=body,
            attachments=[tmp_path],
        )
        print("  Sent OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1
    finally:
        tmp_path.unlink(missing_ok=True)
    print()

    # 3. Verify delivery and attachment
    print("[3/4] Waiting for delivery, then searching again...")
    import time
    time.sleep(5)

    # Use the existing API — search_emails with subject_regex does everything we need
    matched = search_emails(
        subject_regex=re.compile(re.escape(subject), re.IGNORECASE),
    )

    if not matched:
        print("  FAILED: no matching email found after sending")
        return 1

    print(f"  Found {len(matched)} matching email(s)")
    uid = matched[-1]  # most recent

    # Fetch and save attachments to a temp dir
    temp_dir = Path("/tmp/peteos-attachments")
    temp_dir.mkdir(exist_ok=True)
    info = fetch_email(uid=uid, attachments_dir=temp_dir)
    print(f"  From:    {info.from_addr}")
    print(f"  Subject: {info.subject}")
    print(f"  Body:    {info.body_text}")
    print(f"  Attachments: {info.attachments}")

    if subject not in info.subject:
        print(f"  FAILED: expected subject '{subject}'")
        return 1

    if not info.attachments:
        print("  FAILED: no attachments found")
        return 1

    # 4. Verify attachment integrity
    print("[4/4] Verifying attachment integrity...")
    saved = info.attachments[0]
    if not saved.name == "peteos-healthcheck-attachment.txt":
        print(f"  FAILED: unexpected attachment filename {saved.name}")
        return 1
    if saved.read_text() != "Hello from PetEOS — health check attachment":
        print("  FAILED: attachment content mismatch")
        return 1
    print("  Attachment OK")

    # Cleanup
    for child in temp_dir.iterdir():
        child.unlink()
    try:
        temp_dir.rmdir()
    except FileNotFoundError:
        pass

    print()
    print("=== PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
