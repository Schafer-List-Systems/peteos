"""Email: send via SMTP, receive via IMAP.

Zero dependencies — stdlib only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from imaplib import IMAP4_SSL
from pathlib import Path
from smtplib import SMTP
from typing import Sequence

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Send (SMTP)
# ---------------------------------------------------------------------------

def send_email(
    to: str,
    subject: str,
    body: str,
    cc: str | None = None,
    attachments: Sequence[str | Path] = (),
) -> None:
    """Send a plain-text email over TLS.

    Args:
        to: Recipient address.
        subject: Email subject line.
        body: Plain-text body.
        cc: Optional CC address.
        attachments: File paths to attach.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.FROM_ADDRESS
    msg["To"] = to
    if cc:
        msg["Cc"] = cc

    if attachments:
        msg.add_alternative(body, subtype="html")
        for path in attachments:
            msg.add_attachment(
                Path(path).read_bytes(),
                maintype="application",
                subtype="octet-stream",
                filename=Path(path).name,
            )
    else:
        msg.set_content(body)

    with SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.starttls()
        server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
        server.send_message(msg)


# ---------------------------------------------------------------------------
# Receive (IMAP)
# ---------------------------------------------------------------------------

@dataclass
class EmailInfo:
    """Parsed email with extracted content and saved attachments."""

    from_addr: str
    subject: str
    body_text: str
    attachments: list[Path] = field(default_factory=list)


def search_emails(
    folder: str | None = None,
    subject_regex: re.Pattern | None = None,
) -> list[int]:
    """Search IMAP for unread emails matching a subject regex.

    IMAP's UNSEEN flag handles deduplication — once an email is fetched,
    it is marked SEEN on the server and won't appear again.

    Args:
        folder: IMAP folder to search (default from config).
        subject_regex: If provided, filter by subject pattern.

    Returns:
        List of message UIDs that match.
    """
    folder = folder or config.IMAP_FOLDER

    con = IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORT)
    con.login(config.IMAP_USERNAME, config.IMAP_PASSWORD)
    con.select(folder, readonly=True)

    status, data = con.search(None, "UNSEEN")
    if status != "OK":
        logger.warning("IMAP search failed: %s", data)
        con.logout()
        return []

    uids = [int(uid) for uid in data[0].split()]

    if subject_regex is None:
        con.logout()
        return uids

    # Fetch subjects for filtering (batch fetch)
    uid_str = b",".join(str(uid).encode() for uid in uids)
    status, msg_data = con.fetch(uid_str, "(BODY.PEEK[HEADER.FIELDS (SUBJECT)])")
    con.logout()

    if status != "OK":
        return uids

    matched: list[int] = []
    for response in msg_data:
        if not isinstance(response, tuple):
            continue
        subject = _extract_subject_from_header(response[1])
        if subject and subject_regex.search(subject):
            uid = int(response[0].split()[0].replace(b"*", b"").decode())
            matched.append(uid)

    return matched


def fetch_email(
    uid: int,
    folder: str | None = None,
    attachments_dir: Path | None = None,
) -> EmailInfo:
    """Download and parse a single email by UID.

    Args:
        uid: IMAP message UID.
        folder: IMAP folder (default from config).
        attachments_dir: Directory to save attachments.

    Returns:
        Parsed EmailInfo with body text and saved attachments.
    """
    folder = folder or config.IMAP_FOLDER

    con = IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORT)
    con.login(config.IMAP_USERNAME, config.IMAP_PASSWORD)
    con.select(folder)

    status, msg_data = con.fetch(str(uid).encode(), "(RFC822)")
    con.logout()

    if status != "OK" or not msg_data or not msg_data[0]:
        raise ValueError(f"Failed to fetch email UID {uid}")

    raw_bytes = msg_data[0][1]
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    body_text = ""
    attachments: list[Path] = []

    for part in msg.walk():
        content_disposition = part.get("Content-Disposition", "")
        if content_disposition and "attachment" in content_disposition:
            payload = part.get_payload(decode=True)
            if payload and attachments_dir:
                filename = part.get_filename()
                save_path = attachments_dir / filename
                save_path.write_bytes(payload)
                attachments.append(save_path)
        elif part.get_content_type() == "text/plain" and not attachments:
            payload = part.get_payload(decode=True)
            if payload:
                body_text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")

    return EmailInfo(
        from_addr=msg["From"] or "unknown",
        subject=msg["Subject"] or "no subject",
        body_text=body_text,
        attachments=attachments,
    )


def _extract_subject_from_header(raw_header: bytes) -> str | None:
    """Quickly extract Subject from raw header bytes."""
    text = raw_header.decode("utf-8", errors="replace").lower()
    for line in text.split("\n"):
        if line.startswith("subject:"):
            return line.split(":", 1)[1].strip()
    return None