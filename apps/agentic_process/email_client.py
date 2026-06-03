"""Email: send via SMTP, receive via IMAP.

Zero dependencies — stdlib only.
"""

from __future__ import annotations

import logging
import re
import sys
import time
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
    retries: int = 3,
    delay: float = 2.0,
) -> None:
    """Send a plain-text email over TLS.

    Args:
        to: Recipient address.
        subject: Email subject line.
        body: Plain-text body.
        cc: Optional CC address.
        attachments: File paths to attach.
        retries: Number of retry attempts on failure.
        delay: Seconds between retries.
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

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
                server.starttls()
                server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
                server.send_message(msg)
            return
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)

    raise last_err  # type: ignore[misc]


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


def _parse_email_bytes(raw: bytes, save_dir: Path | None = None) -> EmailInfo:
    """Parse raw MIME bytes and return an EmailInfo dataclass.

    Args:
        raw: Raw RFC822 message bytes.
        save_dir: If provided, save attachments to this directory
            and include full paths in the result.
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    body_text = ""
    attachments: list[Path] = []
    for part in msg.walk():
        content_disposition = part.get("Content-Disposition", "")
        if content_disposition and "attachment" in content_disposition:
            payload = part.get_payload(decode=True)
            if payload:
                filename = part.get_filename()
                if save_dir is not None:
                    save_path = save_dir / filename
                    save_path.write_bytes(payload)
                    attachments.append(save_path)
        elif part.get_content_type() == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                body_text = payload.decode(
                    part.get_content_charset() or "utf-8",
                    errors="replace",
                )
    return EmailInfo(
        from_addr=msg["From"] or "unknown",
        subject=msg["Subject"] or "no subject",
        body_text=body_text,
        attachments=attachments,
    )


def _search_one(con, folder: str, subject_regex: re.Pattern | None) -> list[int]:
    """Single attempt to search IMAP (assumes connection is already open)."""
    status, data = con.search(None, "UNSEEN")
    if status != "OK":
        logger.warning("IMAP search failed: %s", data)
        return []

    uids = [int(uid) for uid in data[0].split()]

    if subject_regex is None:
        return uids

    # Fetch subjects for filtering (batch fetch)
    uid_str = b",".join(str(uid).encode() for uid in uids)
    status, msg_data = con.fetch(uid_str, "(BODY.PEEK[HEADER.FIELDS (SUBJECT)])")

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


def search_emails(
    folder: str | None = None,
    subject_regex: re.Pattern | None = None,
    retries: int = 3,
    delay: float = 5.0,
) -> list[int]:
    """Search IMAP for unread emails matching a subject regex.

    IMAP's UNSEEN flag handles deduplication — once an email is fetched,
    it is marked SEEN on the server and won't appear again.

    Args:
        folder: IMAP folder to search (default from config).
        subject_regex: If provided, filter by subject pattern.
        retries: Number of retry attempts on connection failure.
        delay: Seconds between retries.

    Returns:
        List of message UIDs that match.

    Raises:
        SystemExit: After all retries exhausted.
    """
    folder = folder or config.IMAP_FOLDER

    last_err: Exception | None = None
    for attempt in range(retries):
        con: IMAP4_SSL | None = None
        try:
            con = IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORT)
            con.login(config.IMAP_USERNAME, config.IMAP_PASSWORD)
            con.select(folder, readonly=True)

            uids = _search_one(con, folder, subject_regex)

            con.logout()
            return uids

        except Exception as e:
            last_err = e
            if con is not None:
                try:
                    con.logout()
                except Exception:
                    pass
            if attempt < retries - 1:
                time.sleep(delay)

    logger.error("IMAP search failed after %d retries: %s", retries, last_err)
    sys.exit(str(last_err))


def fetch_email(
    uid: int,
    folder: str | None = None,
    cached_inbox: Path | None = None,
    retries: int = 3,
    delay: float = 5.0,
) -> EmailInfo:
    """Download and parse a single email by UID.

    Optionally caches the raw RFC822 message and attachments to
    ``cached_inbox/{uid}/uid.eml`` and ``cached_inbox/{uid}/filename.ext``.

    Args:
        uid: IMAP message UID.
        folder: IMAP folder (default from config).
        cached_inbox: Parent directory for cached emails. Per-email dirs
            are created automatically as ``{cached_inbox}/{uid}/``.
        retries: Number of retry attempts on connection failure.
        delay: Seconds between retries.

    Returns:
        Parsed EmailInfo with body text and attachment paths.

    Raises:
        SystemExit: After all retries exhausted.
    """
    folder = folder or config.IMAP_FOLDER

    last_err: Exception | None = None
    result_data: bytes | None = None
    for attempt in range(retries):
        con: IMAP4_SSL | None = None
        try:
            con = IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORT)
            con.login(config.IMAP_USERNAME, config.IMAP_PASSWORD)
            con.select(folder)

            status, msg_data = con.fetch(str(uid).encode(), "(RFC822)")
            con.logout()
            con = None  # already logged out

            if status != "OK" or not msg_data or not msg_data[0]:
                raise ValueError(f"Failed to fetch email UID {uid}")

            result_data = msg_data[0][1]
            break

        except Exception as e:
            last_err = e
            if con is not None:
                try:
                    con.logout()
                except Exception:
                    pass
            if attempt < retries - 1:
                time.sleep(delay)

    if result_data is None:
        logger.error("IMAP fetch failed after %d retries: %s", retries, last_err)
        sys.exit(str(last_err))

    # Cache raw RFC822 message
    if cached_inbox is not None:
        email_dir = cached_inbox / str(uid)
        email_dir.mkdir(parents=True, exist_ok=True)
        (email_dir / f"{uid}.eml").write_bytes(result_data)

    email_dir = cached_inbox / str(uid) if cached_inbox is not None else None
    return _parse_email_bytes(result_data, save_dir=email_dir)


def get_cached_email(
    uid: int,
    cached_inbox: Path,
    fetch_on_miss: bool = False,
) -> EmailInfo:
    """Return a cached email, optionally fetching it from the server on miss.

    If the email exists in ``cached_inbox/{uid}/{uid}.eml``, parses and
    returns it as ``EmailInfo``.  If the email is not found and
    ``fetch_on_miss`` is ``True``, delegates to ``fetch_email`` which
    downloads and caches it.

    Args:
        uid: IMAP message UID.
        cached_inbox: The cached_inbox parent directory.
        fetch_on_miss: If True, call fetch_email when the cached email
            is not found.  Defaults to False.

    Returns:
        Parsed EmailInfo for the email.
    """
    email_dir = cached_inbox / str(uid)
    eml_path = email_dir / f"{uid}.eml"
    if eml_path.is_file():
        return _parse_email_bytes(eml_path.read_bytes(), save_dir=email_dir)

    if fetch_on_miss:
        return fetch_email(uid, cached_inbox=cached_inbox)

    raise FileNotFoundError(f"Cached email UID {uid} not found")


def list_cached_emails(cached_inbox: Path) -> list[tuple[int, Path]]:
    """List all cached emails in a cached_inbox directory.

    Scans the directory for subdirectories containing a ``{uid}.eml``
    file and returns them as (UID, directory_path) pairs.

    Args:
        cached_inbox: The parent directory containing UID subdirectories.

    Returns:
        List of (uid, cache_dir) tuples sorted by uid ascending.
    """
    if not cached_inbox.is_dir():
        return []

    result: list[tuple[int, Path]] = []
    for entry in sorted(cached_inbox.iterdir()):
        if not entry.is_dir():
            continue
        eml = entry / f"{entry.name}.eml"
        if eml.is_file():
            result.append((int(entry.name), entry))
    return result


def _extract_subject_from_header(raw_header: bytes) -> str | None:
    """Quickly extract Subject from raw header bytes."""
    text = raw_header.decode("utf-8", errors="replace").lower()
    for line in text.split("\n"):
        if line.startswith("subject:"):
            return line.split(":", 1)[1].strip()
    return None