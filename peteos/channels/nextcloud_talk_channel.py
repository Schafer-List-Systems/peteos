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
        nextcloud_url: str,
        bot_id: str,
        bot_secret: str,
        default_role: str = "test",
        host: str = "0.0.0.0",
        port: int = 8766,
    ):
        super().__init__(name, agent)
        self._nextcloud_url = nextcloud_url.rstrip("/")
        self._bot_id = bot_id
        self._bot_secret = bot_secret
        self._default_role = default_role
        self._host = host
        self._port = port
        self._running = False
        self._app: web.Application = None
        self._runner: web.AppRunner = None
        self._site: web.TCPSite = None
        self._server_url: str = ""
        self._rooms: dict[str, uuid.UUID] = {}  # conversation_token -> session_uuid
        self._session_conversations: dict[uuid.UUID, str] = {}  # session_uuid -> conversation_token
        self._incoming_message_ids: dict[uuid.UUID, str] = {}  # session_uuid -> message_id
        self._replied_message_ids: set[str] = set()  # message IDs that received checkmark reaction

    async def start(self) -> str:
        """Start the webhook receiver server.

        Returns:
            The server URL to register as the webhook endpoint with Nextcloud.
        """
        self._app = web.Application()
        self._app.router.add_post("/nextcloud-talk-webhook", self._handle_webhook)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        actual_port = self._site._server.sockets[0].getsockname()[1]
        self._server_url = f"http://{self._host}:{actual_port}/nextcloud-talk-webhook"
        self._running = True

        return self._server_url

    async def stop(self) -> None:
        """Stop the webhook receiver server."""
        self._running = False
        for task in self._session_consumer_tasks.values():
            if task and not task.done():
                task.cancel()
        self._session_consumer_tasks.clear()
        if self._runner:
            await self._runner.cleanup()

    def send(self, message: Message) -> None:
        """Send a message to the originating Nextcloud conversation.

        Iterates over content parts, formats each into text, and sends via
        the Nextcloud Bot API. Skips parts filtered by enable/disable flags.
        For final-answer messages (text-only assistant), sends a checkmark
        reaction after the message text is delivered.

        Args:
            message: The Message to send.
        """
        if not self._active_session_uuid:
            return
        conversation_token = self._session_conversations.get(self._active_session_uuid)
        if not conversation_token:
            return

        if not message.content:
            return

        payloads = []
        for part in message.content:
            if not self._should_send_part(part, message.role):
                continue
            payloads.append(self._format_for_nextcloud(part, message))

        is_final_answer = self._is_final_answer(message)
        asyncio.create_task(self._send_all_sequentially(conversation_token, payloads, is_final_answer))

    def _is_final_answer(self, message: Message) -> bool:
        """Check if a message is a final answer (text-only assistant response).

        A final answer is an assistant message whose only sendable content
        part is text (no tool calls, tool results, reasoning, etc.).

        Args:
            message: The Message to evaluate.

        Returns:
            True if this is a final answer message.
        """
        if message.role != "assistant":
            return False
        for part in message.content:
            if part.type == "text":
                continue
            if self._should_send_part(part, message.role):
                return False
        return True

    def _should_send_part(self, part: ContentPart, msg_role: str) -> bool:
        """Check if a content part should be sent based on enabled/disabled flags."""
        if part.type == "reasoning" and not self._show_reasoning:
            return False
        if part.type == "tool_calls" and not self._show_tool_calls:
            return False
        if part.type == "tool_call" and not self._show_tool_calls:
            return False
        if part.type == "tool_result" and not self._show_tool_results:
            return False
        return True

    def _format_for_nextcloud(self, part: ContentPart, message: Message) -> dict:
        """Format a content part into a Nextcloud-compatible payload.

        All messages are rendered as Markdown by the Nextcloud Talk API.
        """
        payload = {
            "message": "",
            "replyTo": "",
            "referenceId": message.id,
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
                payload["message"] = f"```python\n{tc.get('name', '?')}({args})\n```"

            elif isinstance(tc, list):
                blocks = []
                for item in tc:
                    args = item.get("arguments", "{}")
                    blocks.append(f"{item.get('name', '?')}({args})")
                payload["message"] = "\n".join(
                    f"```python\n{block}\n```" for block in blocks
                )

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

    def receive(self) -> str | None:
        """Webhook-driven channel - no polling.

        Returns:
            None always.
        """
        return None

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
            self._bot_secret.encode(),
            random_nonce.encode() + body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(computed, signature)

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

        session = self._get_or_create_session(conversation_token)
        if session:
            self._active_session_uuid = session.uuid
            message_id = obj.get("id")
            if message_id:
                self._incoming_message_ids[session.uuid] = message_id

            user_message = Message(
                role="user",
                content=[ContentPart(part_type="text", text=message_text)],
            )
            await session.queue_message(user_message)

            # Send thinking reaction to the incoming message
            if message_id:
                await self._send_reaction(conversation_token, message_id, "🤔")

    async def _handle_reaction(self, event: dict) -> None:
        """Handle reaction added (Like event)."""
        content = event.get("content", "")
        actor = event.get("actor", {})
        display_name = actor.get("displayName", "unknown")
        logger.info("Reaction '%s' by %s", content, display_name)

    async def _handle_reaction_undo(self, event: dict) -> None:
        """Handle reaction removed (Undo event)."""
        content = event.get("object", {}).get("content", "")
        actor = event.get("actor", {})
        display_name = actor.get("displayName", "unknown")
        logger.info("Reaction removed '%s' by %s", content, display_name)

    async def _handle_join(self, event: dict) -> None:
        """Handle bot added to room (Join event)."""
        obj = event.get("object", {})
        # conversation_token may be in target.id or object.token
        target = event.get("target", {})
        conversation_token = target.get("id", "") or obj.get("token", "")
        actor = event.get("actor", {})
        display_name = actor.get("displayName", actor.get("name", actor.get("id", "unknown")))
        logger.info("Bot added to room by %s, conversation=%s", display_name, conversation_token)

        session = self._get_or_create_session(conversation_token)
        if session:
            self._active_session_uuid = session.uuid

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

    def _get_or_create_session(self, conversation_token: str):
        """Get existing session or create a new one for a conversation.

        Args:
            conversation_token: Nextcloud Talk conversation token.

        Returns:
            The session object, or None if creation fails.
        """
        if conversation_token in self._rooms:
            session_uuid = self._rooms[conversation_token]
            session = self._agent.get_session(session_uuid)
            self._subscribe_session(session_uuid)
            return session

        try:
            session = self._agent.create_session(self._default_role)
            self._rooms[conversation_token] = session.uuid
            self._session_conversations[session.uuid] = conversation_token
            self._subscribe_session(session.uuid)
            logger.info("Created session %s for conversation %s", session.uuid, conversation_token)
            return session
        except Exception as e:
            logger.error("Failed to create session for conversation %s: %s", conversation_token, e)
            return None

    def _subscribe_session(self, session_uuid: uuid.UUID) -> None:
        """Subscribe this channel to receive notifications for a session.

        Delegates to the base class which handles queue creation, channel
        registration, and notification consumer lifecycle.

        Args:
            session_uuid: The session to subscribe to.
        """
        self.subscribe_to_session(session_uuid)

    async def _send_to_nextcloud(self, conversation_token: str, payload: dict) -> None:
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
        """
        try:
            import aiohttp

            json_body = json.dumps(payload)
            random_nonce = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
            # Sign random + message text (same as official bash example)
            signature = hmac.new(
                self._bot_secret.encode(),
                (random_nonce + payload["message"]).encode(),
                hashlib.sha256,
            ).hexdigest()

            url = (
                f"{self._nextcloud_url}/ocs/v2.php/apps/spreed/api/v1/bot/{conversation_token}/message"
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
                        logger.debug("Message sent to conversation %s", conversation_token)
                    else:
                        body = await resp.text()
                        logger.warning(
                            "Failed to send message to %s: %d %s",
                            conversation_token,
                            resp.status,
                            body,
                        )

        except Exception as e:
            logger.error("Error sending to Nextcloud: %s", e)

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
                self._bot_secret.encode(),
                (random_nonce + emoji).encode(),
                hashlib.sha256,
            ).hexdigest()

            url = (
                f"{self._nextcloud_url}/ocs/v2.php/apps/spreed/api/v1/bot/{conversation_token}/reaction/{message_id}"
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
        self, conversation_token: str, payloads: list[dict], is_final_answer: bool = False
    ) -> None:
        """Send multiple payloads to Nextcloud sequentially in order.

        Args:
            conversation_token: The conversation to send to.
            payloads: List of formatted message payloads to send.
            is_final_answer: If True, sends a checkmark reaction after
                the last message to indicate the incoming user message
                has been answered.
        """
        for i, payload in enumerate(payloads):
            await self._send_to_nextcloud(conversation_token, payload)

        if is_final_answer and self._active_session_uuid:
            message_id = self._incoming_message_ids.get(self._active_session_uuid)
            if message_id and message_id not in self._replied_message_ids:
                self._replied_message_ids.add(message_id)
                await self._send_reaction(conversation_token, message_id, "🤖")
