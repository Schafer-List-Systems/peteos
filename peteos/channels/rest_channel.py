import asyncio
import json
import uuid

import aiohttp
from aiohttp import web

from peteos.channel import Channel
from peteos.chatbot import Message, ContentPart


class RESTApiChannel(Channel):
    """REST API channel for remote agent interaction."""

    def __init__(self, name: str, agent, host: str = "127.0.0.1", port: int = 8080):
        """
        Initialize RESTApiChannel.

        Args:
            name: Unique identifier for this channel.
            agent: The Agent instance this channel connects to.
            host: Server host to bind to.
            port: Server port to listen on.
        """
        super().__init__(name, agent)
        self._host = host
        self._port = port
        self._running = False
        self._app: web.Application = None
        self._runner: web.AppRunner = None
        self._site: web.TCPSite = None
        self._server_url: str = ""
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscriptions: dict[uuid.UUID, asyncio.Queue] = {}

    async def start(self) -> str:
        """Start the REST API server.

        Returns:
            The server URL.
        """
        self._app = web.Application()
        self._app.router.add_post("/chat", self._handle_chat)
        self._app.router.add_post("/sessions", self._handle_create_session)
        self._app.router.add_get("/sessions", self._handle_list_sessions)
        self._app.router.add_post("/sessions/{session_uuid}/select", self._handle_select_session)
        self._app.router.add_get("/sessions/{session_uuid}/messages", self._handle_get_messages)
        self._app.router.add_get("/ws/{session_uuid}", self._handle_websocket)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        # Get the actual port assigned (especially when port=0)
        actual_port = self._site._server.sockets[0].getsockname()[1]
        self._server_url = f"http://{self._host}:{actual_port}"
        self._running = True

        return self._server_url

    async def stop(self) -> None:
        """Stop the REST API server."""
        self._running = False
        self._message_queue.put_nowait(None)

        # Close all WebSocket connections
        for queue in list(self._subscriptions.values()):
            queue.put_nowait(None)

        if self._runner:
            await self._runner.cleanup()

    def send(self, message: str) -> None:
        """Send a message through the WebSocket stream.

        Args:
            message: The message to send.
        """
        # Find all subscriptions for the active session
        if self._active_session_uuid:
            queue = self._subscriptions.get(self._active_session_uuid)
            if queue:
                asyncio.create_task(queue.put(json.dumps({"type": "message", "content": message})))

    def receive(self) -> str | None:
        """Receive a message from the queue (async context required).

        Returns:
            The received message, or None if channel is closed.
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._message_queue.get())
        except RuntimeError:
            # No event loop running, return None
            return None

    async def _handle_chat(self, request: web.Request) -> web.Response:
        """Handle POST /chat - send message to active session."""
        if self._active_session_uuid is None:
            return web.json_response({"error": "No session selected"}, status=400)

        try:
            data = await request.json()
            content = data.get("content", "")

            session = self._agent.get_session(self._active_session_uuid)
            if session is None:
                return web.json_response({"error": "Session not found"}, status=404)

            message = Message(
                role="user",
                content=[ContentPart(part_type="text", text=content)]
            )
            await session.queue_message(message)

            return web.json_response({"status": "message_queued"})

        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

    async def _handle_create_session(self, request: web.Request) -> web.Response:
        """Handle POST /sessions - create a new session."""
        try:
            data = await request.json()
            role_name = data.get("role", "")

            if not role_name:
                return web.json_response({"error": "role is required"}, status=400)

            session = self._agent.create_session(role_name)

            return web.json_response({
                "status": "created",
                "uuid": str(session.uuid)
            })

        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _handle_list_sessions(self, request: web.Request) -> web.Response:
        """Handle GET /sessions - list all sessions."""
        sessions = self._agent.list_sessions()
        result = []

        for session_uuid, session in sessions.items():
            result.append({
                "uuid": str(session_uuid),
                "role": session.role.name,
                "is_active": session_uuid == self._active_session_uuid
            })

        return web.json_response({"sessions": result})

    async def _handle_select_session(self, request: web.Request) -> web.Response:
        """Handle POST /sessions/{uuid}/select - select active session."""
        try:
            session_uuid = uuid.UUID(request.match_info["session_uuid"])

            if self._agent.get_session(session_uuid) is None:
                return web.json_response({"error": "Session not found"}, status=404)

            self.select_session(session_uuid)

            return web.json_response({
                "status": "selected",
                "uuid": str(session_uuid)
            })

        except ValueError:
            return web.json_response({"error": "Invalid UUID"}, status=400)

    async def _handle_get_messages(self, request: web.Request) -> web.Response:
        """Handle GET /sessions/{uuid}/messages - get session messages."""
        try:
            session_uuid = uuid.UUID(request.match_info["session_uuid"])
            session = self._agent.get_session(session_uuid)

            if session is None:
                return web.json_response({"error": "Session not found"}, status=404)

            history = session.chat_history.messages
            result = []

            for msg in history:
                content = msg.content
                result.append({
                    "role": content.get("role", "unknown"),
                    "content": content.get("text", content.get("content", "")),
                    "timestamp": msg.creation_timestamp.isoformat()
                })

            return web.json_response({"messages": result})

        except ValueError:
            return web.json_response({"error": "Invalid UUID"}, status=400)

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WS /ws/{session_uuid} - WebSocket connection for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        session_uuid = uuid.UUID(request.match_info["session_uuid"])
        self._subscriptions[session_uuid] = ws

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "chat":
                        if self._active_session_uuid is None:
                            await ws.send_json({"error": "No session selected"})
                            continue

                        session = self._agent.get_session(self._active_session_uuid)
                        if session:
                            message = Message(
                                role="user",
                                content=[ContentPart(part_type="text", text=data.get("content", ""))]
                            )
                            await session.queue_message(message)

                    elif data.get("type") == "subscribe":
                        # Re-subscribe to a different session
                        target_uuid = uuid.UUID(data.get("session_uuid", ""))
                        if self._agent.get_session(target_uuid):
                            self._subscriptions[target_uuid] = ws
                            await ws.send_json({"status": "subscribed", "session_uuid": str(target_uuid)})

        except Exception as e:
            pass
        finally:
            self._subscriptions.pop(session_uuid, None)

        return ws
