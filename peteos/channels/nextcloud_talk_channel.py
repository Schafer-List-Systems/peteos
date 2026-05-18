"""NextcloudTalkChannel - Webhook-driven bot channel for Nextcloud Talk."""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from typing import Optional

from aiohttp import web

from peteos.channels.channel import Channel
from peteos.chatbot import Message, ContentPart
from peteos.session import ApprovalEvent, ToolCallRecord, ToolApprovalStatus, ToolExecutionStatus

logger = logging.getLogger(__name__)


class NextcloudTalkChannel(Channel):
    """Nextcloud Talk bot channel for webhook-driven agent interaction.

    Usage in Nextcloud:
        1. Install the bot: ./occ talk:bot:install <name> <webhook-url> <secret>
        2. Enable features (required for webhooks to work):
           ./occ talk:bot:state <bot-id> 1 --feature webhook --feature response --feature reaction
        3. Start this channel and register the webhook URL from its output.
    """

    def __init__(
        self,
        name: str,
        agent,
        config: dict,
    ):
        super().__init__(name, agent)
        self._config = config
        self._app: web.Application = None
        self._runner: web.AppRunner = None
        self._site: web.TCPSite = None
        self._server_url: str = ""
        self._on_room_joined: callable | None = None
        self._rooms: dict[str, uuid.UUID] = {}  # conversation_token -> session_uuid
        self._session_conversations: dict[uuid.UUID, str] = {}  # session_uuid -> conversation_token
        self._incoming_message_ids: dict[uuid.UUID, str] = {}  # session_uuid -> message_id
        self._replied_message_ids: set[str] = set()  # message IDs that received checkmark reaction
        self._tool_call_ids: dict[str, list[str]] = {}  # referenceId -> [tool_call_ids]
        self._sent_message_sessions: dict[str, uuid.UUID] = {}  # referenceId -> session_uuid
        self._sent_messages: list[dict] = []  # Local history: [{referenceId, message, tool_call_ids, session_uuid}]

    @staticmethod
    def load_config(config_file: str = "examples/config/nextcloud_config.json") -> dict:
        """Load and validate Nextcloud config from JSON file.

        Required fields: nextcloud_url, bot_id, bot_secret.
        Optional fields with defaults: default_role, host, port,
        show_reasoning, show_tool_calls, show_tool_results.

        Args:
            config_file: Path to the JSON configuration file.

        Returns:
            Dict with validated config values and defaults applied.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            KeyError: If required fields are missing.
        """
        with open(config_file, "r") as f:
            config = json.load(f)
        config.setdefault("default_role", "test")
        config.setdefault("host", "0.0.0.0")
        config.setdefault("port", 8766)
        config.setdefault("show_reasoning", True)
        config.setdefault("show_tool_calls", True)
        config.setdefault("show_tool_results", True)
        config.setdefault("prefix_actor_names", False)
        required = ["nextcloud_url", "bot_id", "bot_secret"]
        missing = [k for k in required if k not in config]
        if missing:
            raise KeyError(f"Missing required config fields: {', '.join(missing)}")
        return config

    async def start(self) -> str:
        """Start the webhook receiver server.

        Returns:
            The server URL to register as the webhook endpoint with Nextcloud.
        """
        # Start ActiveClass's notification consumption loop
        await super().start()

        self._app = web.Application()
        self._app.router.add_post("/nextcloud-talk-webhook", self._handle_webhook)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self._config["host"], self._config["port"])
        await self._site.start()

        actual_port = self._site._server.sockets[0].getsockname()[1]
        self._server_url = f"http://{self._config['host']}:{actual_port}/nextcloud-talk-webhook"

        return self._server_url

    async def stop(self) -> None:
        """Stop the webhook receiver server."""
        self._running = False
        if self._runner:
            await self._runner.cleanup()

    async def send(self, message: Message, session_uuid: uuid.UUID | None = None) -> None:
        """Send a message to the originating Nextcloud conversation.

        Iterates over content parts, formats each into text, and sends via
        the Nextcloud Bot API. Skips parts filtered by enable/disable flags.
        For final-answer messages (text-only assistant), sends a checkmark
        reaction after the message text is delivered.

        Muted messages (``_sent_muted`` in metadata) are sent as empty
        messages so reactions still fire, but no actual text payload goes
        to the user.

        Args:
            message: The Message to send.
            session_uuid: The session UUID to route to.
        """
        if not session_uuid:
            logger.debug("[nextcloud] send(): NO session_uuid, dropping %s %s", message.get_role(), message.get_id()[:8])
            return
        conversation_token = self._session_conversations.get(session_uuid)
        if not conversation_token:
            logger.debug("[nextcloud] send(): NO conversation_token for %s, dropping %s", session_uuid, message.get_id()[:8])
            return

        if not message.content:
            logger.debug("[nextcloud] send(): NO content for %s, dropping %s", message.get_role(), message.get_id()[:8])
            return

        payloads = []
        is_muted = message.metadata.get("_sent_muted", False)
        logger.debug("[nextcloud] send(): %s id=%s muted=%s parts=%d", message.get_role(), message.get_id()[:8], is_muted, len(message.content))
        for part in message.content:
            if is_muted:
                # Skip all content parts for muted messages — reactions fire via _send_all_sequentially
                continue
            if not self._should_send_part(part, message.get_role()):
                continue
            payload = self._format_for_nextcloud(part, message.get_id())
            # Extract tool_call_ids from tool_call/tool_calls content parts
            # so _send_all_sequentially can track the mapping
            if part.type in ("tool_call", "tool_calls", "tool_use"):
                tool_call_ids: list[str] = []
                if part.type == "tool_use":
                    # Anthropic format: id is a direct key
                    id_val = part.data.get("id", "")
                    if id_val:
                        tool_call_ids.append(id_val)
                else:
                    tc = part.data.get("tool_call") or part.data.get("tool_calls")
                    if isinstance(tc, dict):
                        id_val = tc.get("id", "")
                        if id_val:
                            tool_call_ids.append(id_val)
                    elif isinstance(tc, list):
                        tool_call_ids = [item.get("id", "") for item in tc if item.get("id", "")]
                if tool_call_ids:
                    payload["tool_call_ids"] = tool_call_ids
            payloads.append(payload)

        is_final_answer = message.metadata.get("finish", False)
        await self._send_all_sequentially(conversation_token, payloads, is_final_answer, session_uuid)

    def _should_send_part(self, part: ContentPart, msg_role: str) -> bool:
        """Check if a content part should be sent based on enabled/disabled flags."""
        if part.type == "reasoning" and not self._config.get("show_reasoning", True):
            return False
        if part.type == "tool_calls" and not self._config.get("show_tool_calls", True):
            return False
        if part.type == "tool_call" and not self._config.get("show_tool_calls", True):
            return False
        if part.type == "tool_result" and not self._config.get("show_tool_results", True):
            return False
        return True

    def _format_for_nextcloud(self, part: ContentPart, reference_id: str) -> dict:
        """Format a content part into a Nextcloud-compatible payload.

        All messages are rendered as Markdown by the Nextcloud Talk API.
        """
        payload = {
            "message": "",
            "replyTo": "",
            "referenceId": reference_id,
            "silent": part.type != "text",
        }
        if part.type == "text":
            payload["message"] = part.data.get("text", "")

        elif part.type == "reasoning":
            reasoning = part.data.get("reasoning", "")
            payload["message"] = f"> _{reasoning}_"

        elif part.type in ("tool_calls", "tool_call"):
            tc = part.data.get("tool_call") or part.data.get("tool_calls")

            if isinstance(tc, dict):
                args = tc.get("arguments", "{}")
                tool_id = tc.get("id", "?")
                payload["message"] = f"```python\n{tc.get('name', '?')}({args})\n```\n/* id: {tool_id} */"

            elif isinstance(tc, list):
                parts = []
                for item in tc:
                    args = item.get("arguments", "{}")
                    tool_id = item.get("id", "?")
                    parts.append(f"```python\n{item.get('name', '?')}({args})\n```\n/* id: {tool_id} */")
                payload["message"] = "\n".join(parts)

            else:
                payload["message"] = "```python\n(no data)\n```"

        elif part.type == "tool_result":
            content = part.data.get("content", "")
            if isinstance(content, (dict, list)):
                payload["message"] = "```json\n" + json.dumps(content, indent=2) + "\n```"
            elif isinstance(content, str):
                payload["message"] = content
            else:
                payload["message"] = str(content)

        return payload


    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming webhook from Nextcloud Talk.

        Verifies the HMAC-SHA256 signature, parses the Activity Streams 2.0
        payload, and dispatches to the appropriate handler.
        """
        try:
            body = await request.read()
            random_nonce = request.headers.get("X-NEXTCLOUD-TALK-RANDOM", "")
            signature = request.headers.get("X-NEXTCLOUD-TALK-SIGNATURE", "")

            if not self._verify_signature(body, random_nonce, signature):
                logger.warning("Invalid webhook signature from %s", request.remote)
                return web.json_response({"error": "invalid signature"}, status=401)

            event = json.loads(body)
            event_type = event.get("type", "")
            logger.debug("Received %s event", event_type)

            handler = {
                "Create": self._handle_message,
                "Like": self._handle_reaction,
                "Undo": self._handle_reaction_undo,
                "Join": self._handle_join,
                "Leave": self._handle_leave,
            }.get(event_type)

            if handler:
                await handler(event)
            else:
                logger.warning("Unknown event type: %s", event_type)

            return web.json_response({"status": "ok"})

        except Exception as e:
            logger.exception("Error processing webhook")
            return web.json_response({"error": str(e)}, status=500)

    def _verify_signature(self, body: bytes, random_nonce: str, signature: str) -> bool:
        """Verify HMAC-SHA256 webhook signature.

        Args:
            body: Raw request body.
            random_nonce: The X-NEXTCLOUD-TALK-RANDOM header value.
            signature: The X-NEXTCLOUD-TALK-SIGNATURE header value.

        Returns:
            True if signature is valid.
        """
        computed = hmac.new(
            self._config["bot_secret"].encode(),
            random_nonce.encode() + body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(computed, signature)

    async def register_room(self, session_uuid: uuid.UUID, conversation_token: str) -> None:
        """Register a session with a Nextcloud Talk conversation token.

        The app creates sessions and calls this to map them to rooms.
        After registration, incoming messages for this room will be routed
        to the session.

        Args:
            session_uuid: The session to route messages to.
            conversation_token: The Nextcloud Talk conversation token.
        """
        self._rooms[conversation_token] = session_uuid
        self._session_conversations[session_uuid] = conversation_token
        self.subscribe_to_session(session_uuid)

        await self.send(
            Message(role="assistant", content=[ContentPart(part_type="text", text="Hello, I am online now.")]),
            session_uuid=session_uuid,
        )

        logger.info("Registered room %s with session %s", conversation_token, session_uuid)

    async def _handle_message(self, event: dict) -> None:
        """Handle incoming chat message (Create event)."""
        logger.debug("Raw Create event: %s", json.dumps(event, indent=2, default=str))
        actor = event.get("actor", {})
        display_name = actor.get("displayName", actor.get("name", actor.get("id", "unknown")))
        logger.info("Message from %s", display_name)

        obj = event.get("object", {})
        content_str = obj.get("content", "")

        # conversation_token is under target.id, not object.token
        target = event.get("target", {})
        conversation_token = target.get("id", "") or obj.get("token", "")

        # Parse content: may contain mention placeholders like {mention-user1}
        try:
            content = json.loads(content_str) if content_str else {}
            message_text = content.get("message", "")
        except json.JSONDecodeError:
            message_text = content_str

        if not message_text:
            logger.warning("Empty message from %s", display_name)
            return

        session = await self._find_session(conversation_token)
        if session:
            message_id = obj.get("id")
            if message_id:
                self._incoming_message_ids[session.uuid] = message_id
                # Send thinking reaction immediately, before processing
                await self._send_reaction(conversation_token, message_id, "🤔")

            if self._config.get("prefix_actor_names", False):
                message_text = f"User {display_name} wrote: {message_text}"

            user_message = Message(
                role="user",
                content=[ContentPart(part_type="text", text=message_text)],
            )
            await session.queue_message(user_message)

    async def _handle_reaction(self, event: dict) -> None:
        """Handle reaction added (Like event). Approve or deny tool calls."""
        emoji = event.get("content", "")
        if emoji not in ("+1", "-1", "\U0001f44d", "\U0001f44e"):
            return

        obj = event.get("object", {})
        server_message_id = obj.get("id", "")
        # Parse content - can be JSON string or plain string
        reaction_message = ""
        content_raw = obj.get("content", "")
        try:
            if isinstance(content_raw, str):
                content_parsed = json.loads(content_raw)
                reaction_message = content_parsed.get("message", "")
            else:
                reaction_message = content_raw.get("message", "") if isinstance(content_raw, dict) else ""
        except (json.JSONDecodeError, TypeError):
            reaction_message = str(content_raw) if content_raw else ""

        logger.info("Reaction '%s' on message %s: %s", emoji, server_message_id, reaction_message)

        # Match against our local sent message history
        reference_id = None
        for sent in reversed(self._sent_messages):
            if sent.get("message", "") == reaction_message:
                reference_id = sent.get("referenceId", "")
                logger.info("Matched reaction to local message with referenceId %s", reference_id)
                break

        if not reference_id:
            logger.warning("No matching local message found for reaction text")
            return

        tool_call_ids = self._tool_call_ids.get(reference_id, [])
        if not tool_call_ids:
            logger.warning("No tool calls found for matched message %s", reference_id)
            return

        session_uuid = self._sent_message_sessions.get(reference_id)
        if not session_uuid:
            logger.warning("No session found for matched message %s", reference_id)
            return

        session = self._agent.get_session(session_uuid)
        if not session:
            return

        approved = emoji in ("+1", "\U0001f44d")
        for tool_call_id in tool_call_ids:
            record = session._find_pending_record(tool_call_id)
            if record is None:
                logger.warning("No pending record for tool_call_id=%s", tool_call_id)
                continue

            tool_call = record.tool_call
            approval_event = ApprovalEvent(
                tool_call_id=tool_call_id,
                tool_call=tool_call,
                approved=approved,
            )
            session.push_event(approval_event)
            logger.info("Tool call %s %s via reaction %s", tool_call_id, "approved" if approved else "denied", emoji)
            return

    async def _handle_reaction_undo(self, event: dict) -> None:
        """Handle reaction removed (Undo event). Revert approval status."""
        emoji = event.get("object", {}).get("content", "")
        if emoji not in ("+1", "-1", "\U0001f44d", "\U0001f44e"):
            return

        obj = event.get("object", {})
        server_message_id = obj.get("id", "")
        # Parse content - can be JSON string or plain string
        reaction_message = ""
        content_raw = obj.get("content", "")
        try:
            if isinstance(content_raw, str):
                content_parsed = json.loads(content_raw)
                reaction_message = content_parsed.get("message", "")
            else:
                reaction_message = content_raw.get("message", "") if isinstance(content_raw, dict) else ""
        except (json.JSONDecodeError, TypeError):
            reaction_message = str(content_raw) if content_raw else ""

        logger.info("Reaction removed '%s' from message %s: %s", emoji, server_message_id, reaction_message)

        # Match against our local sent message history
        reference_id = None
        for sent in reversed(self._sent_messages):
            if sent.get("message", "") == reaction_message:
                reference_id = sent.get("referenceId", "")
                break

        if not reference_id:
            logger.warning("No matching local message found for reaction")
            return

        tool_call_ids = self._tool_call_ids.get(reference_id, [])
        if not tool_call_ids:
            return

        session_uuid = self._sent_message_sessions.get(reference_id)
        if not session_uuid:
            return

        session = self._agent.get_session(session_uuid)
        if not session:
            return

        for tool_call_id in tool_call_ids:
            record = session._find_pending_record(tool_call_id)
            if record is None:
                continue
            record.approval_status = ToolApprovalStatus.PENDING
            record.execution_status = ToolExecutionStatus.WAITING_FOR_APPROVAL

    async def _handle_join(self, event: dict) -> None:
        """Handle bot added to room (Join event)."""
        obj = event.get("object", {})
        # conversation_token may be in target.id or object.token
        target = event.get("target", {})
        conversation_token = target.get("id", "") or obj.get("token", "")
        actor = event.get("actor", {})
        display_name = actor.get("displayName", actor.get("name", actor.get("id", "unknown")))
        logger.info("Bot added to room by %s, conversation=%s", display_name, conversation_token)

        if self._on_room_joined:
            await self._on_room_joined(conversation_token)

    async def _handle_leave(self, event: dict) -> None:
        """Handle bot removed from room (Leave event)."""
        obj = event.get("object", {})
        target = event.get("target", {})
        conversation_token = target.get("id", "") or obj.get("token", "")
        actor = event.get("actor", {})
        display_name = actor.get("displayName", actor.get("name", actor.get("id", "unknown")))
        logger.info("Bot removed from room by %s, conversation=%s", display_name, conversation_token)

        session_uuid = self._rooms.pop(conversation_token, None)
        if session_uuid:
            self._session_conversations.pop(session_uuid, None)
            self._incoming_message_ids.pop(session_uuid, None)
            self.unsubscribe_from_session(session_uuid)
        logger.info("Removed mapping for conversation %s", conversation_token)

    async def _find_session(self, conversation_token: str):
        """Find a registered session for a conversation token.

        Returns None if the room is not registered (app must call register_room first).

        Args:
            conversation_token: Nextcloud Talk conversation token.

        Returns:
            The session object, or None if room is not registered.
        """
        session_uuid = self._rooms.get(conversation_token)
        if session_uuid:
            return self._agent.get_session(session_uuid)
        return None

    async def _send_to_nextcloud(self, conversation_token: str, payload: dict) -> Optional[str]:
        """Send a message to a Nextcloud Talk conversation.

        Per the official Nextcloud Talk Bots API:
        - Endpoint: POST /ocs/v2.php/apps/spreed/api/v1/bot/{TOKEN}/message
          where TOKEN is the conversation token, NOT the bot ID
        - Content-Type: application/json
        - Body: {"message": "...", "replyTo": ..., "referenceId": ..., "silent": ...}
        - Signature: HMAC-SHA256 of random_header + raw request body

        Args:
            conversation_token: The conversation to send to.
            payload: Dict with message, replyTo, referenceId, silent keys.

        Returns:
            The message ID from the response, or None on failure.
        """
        try:
            import aiohttp

            json_body = json.dumps(payload)
            random_nonce = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
            # Sign random + message text (same as official bash example)
            signature = hmac.new(
                self._config["bot_secret"].encode(),
                (random_nonce + payload["message"]).encode(),
                hashlib.sha256,
            ).hexdigest()

            url = (
                f"{self._config["nextcloud_url"].rstrip("/")}/ocs/v2.php/apps/spreed/api/v1/bot/{conversation_token}/message"
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=json_body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "OCS-APIRequest": "true",
                        "X-Nextcloud-Talk-Bot-Random": random_nonce,
                        "X-Nextcloud-Talk-Bot-Signature": signature,
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 201:
                        await resp.json()
                        logger.debug("Message sent to conversation %s", conversation_token)
                        return None
                    else:
                        body = await resp.text()
                        logger.warning(
                            "Failed to send message to %s: %d %s",
                            conversation_token,
                            resp.status,
                            body,
                        )
                        return None

        except Exception as e:
            logger.error("Error sending to Nextcloud: %s", e)
            return None

    async def _send_reaction(self, conversation_token: str, message_id: str, emoji: str) -> None:
        """Send an emoji reaction to a message in a Nextcloud Talk conversation.

        Per the Nextcloud Talk Bots API:
        - Endpoint: POST /ocs/v2.php/apps/spreed/api/v1/bot/{TOKEN}/reaction/{MESSAGE_ID}
        - Content-Type: application/json
        - Body: {"reaction": "emoji"}
        - Signature: HMAC-SHA256 of random_header + raw request body

        Args:
            conversation_token: The conversation token.
            message_id: The message object.id to react to.
            emoji: The emoji string to use as the reaction.
        """
        try:
            import aiohttp

            # Compact JSON to match bash reference: {"reaction":"❓"} not {"reaction": "❓"}
            json_body = json.dumps({"reaction": emoji}, separators=(",", ":"))
            random_nonce = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
            # Per Nextcloud Talk Bots API bash implementation: HMAC-SHA256 of
            # random + full JSON body
            signature = hmac.new(
                self._config["bot_secret"].encode(),
                (random_nonce + emoji).encode(),
                hashlib.sha256,
            ).hexdigest()

            url = (
                f"{self._config["nextcloud_url"].rstrip("/")}/ocs/v2.php/apps/spreed/api/v1/bot/{conversation_token}/reaction/{message_id}"
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=json_body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "OCS-APIRequest": "true",
                        "X-Nextcloud-Talk-Bot-Random": random_nonce,
                        "X-Nextcloud-Talk-Bot-Signature": signature,
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 201):
                        logger.debug(
                            "Reaction '%s' sent to message %s in conversation %s",
                            emoji,
                            message_id,
                            conversation_token,
                        )
                    else:
                        body = await resp.text()
                        logger.warning(
                            "Failed to send reaction to message %s: %d %s",
                            message_id,
                            resp.status,
                            body,
                        )

        except Exception as e:
            logger.error("Error sending reaction to Nextcloud: %s", e)

    async def _send_all_sequentially(
        self, conversation_token: str, payloads: list[dict], is_final_answer: bool = False, session_uuid: uuid.UUID | None = None
    ) -> None:
        """Send multiple payloads to Nextcloud sequentially in order.

        Args:
            conversation_token: The conversation to send to.
            payloads: List of formatted message payloads to send.
            is_final_answer: If True, sends a checkmark reaction after
                the last message to indicate the incoming user message
                has been answered.
        """
        sent_reference_ids: list[str] = []
        for i, payload in enumerate(payloads):
            _ = await self._send_to_nextcloud(conversation_token, payload)
            # referenceId is set in the payload by _format_for_nextcloud
            reference_id = payload.get("referenceId", "")
            if reference_id:
                sent_reference_ids.append(reference_id)
                # Track referenceId → session for reaction matching
                if session_uuid:
                    self._sent_message_sessions[reference_id] = session_uuid
                # For tool_call/tool_calls type messages, extract tool_call_ids
                tool_calls = payload.get("tool_call_ids", [])
                if tool_calls:
                    self._tool_call_ids[reference_id] = tool_calls
                # Store the formatted message text for local matching
                msg_text = payload.get("message", "")
                self._sent_messages.append({
                    "referenceId": reference_id,
                    "message": msg_text,
                    "tool_call_ids": tool_calls,
                    "session_uuid": session_uuid,
                })

        if is_final_answer and session_uuid:
            message_id = self._incoming_message_ids.get(session_uuid)
            if message_id and message_id not in self._replied_message_ids:
                self._replied_message_ids.add(message_id)
                await self._send_reaction(conversation_token, message_id, "🤖")
