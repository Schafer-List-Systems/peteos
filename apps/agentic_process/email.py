"""Send outbound email via SMTP."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from . import config


def send_email(
    to: str,
    subject: str,
    body: str,
    cc: str | None = None,
) -> None:
    """Send a plain-text email over TLS.

    Args:
        to: Recipient address.
        subject: Email subject line.
        body: Plain-text body.
        cc: Optional CC address.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.FROM_ADDRESS
    msg["To"] = to
    if cc:
        msg["Cc"] = cc
    msg.set_content(body)

    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.starttls()
        server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
        server.send_message(msg)