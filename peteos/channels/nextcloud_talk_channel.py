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

    def send(self, message: str) -> None:
        """Send a message to the originating Nextcloud conversation.

        Looks up the conversation token for the currently active session
        and dispatches the message via the Nextcloud Bot API.

        Args:
            message: The message text to send.
        """
        if not self._active_session_uuid:
            return
        conversation_token = self._session_conversations.get(self._active_session_uuid)
        if not conversation_token:
            return
        asyncio.create_task(self._send_to_nextcloud(conversation_token, message))

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
            user_message = Message(
                role="user",
                content=[ContentPart(part_type="text", text=message_text)],
            )
            await session.queue_message(user_message)

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

    async def _send_to_nextcloud(self, conversation_token: str, message_text: str) -> None:
        """Send a message to a Nextcloud Talk conversation.

        Per the official Nextcloud Talk Bots API:
        - Endpoint: POST /ocs/v2.php/apps/spreed/api/v1/bot/{TOKEN}/message
          where TOKEN is the conversation token, NOT the bot ID
        - Content-Type: application/json
        - Body: {"message": "..."}
        - Signature: HMAC-SHA256 of random_header + raw request body

        Args:
            conversation_token: The conversation to send to.
            message_text: The message text (Markdown supported).
        """
        try:
            import aiohttp

            random_nonce = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
            json_body = json.dumps({"message": message_text})
            # Sign random + raw message text (same as official bash example)
            signature = hmac.new(
                self._bot_secret.encode(),
                (random_nonce + message_text).encode(),
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
