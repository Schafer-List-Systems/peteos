# Nextcloud Talk Bots API

Reference for implementing a Nextcloud Talk channel in the agent system. All information sourced from [official Nextcloud Talk Bots documentation](https://nextcloud-talk.readthedocs.io/en/latest/bots/).

## Overview

Nextcloud Talk bots are webhook-driven automation agents introduced in Talk 17.1. They receive incoming messages from chat rooms and can send messages, reactions, and manage reactions back. Bots are registered per-server via CLI with a shared secret for cryptographic message signing.

## Setup & Installation

Bots can only be installed via CLI for security reasons:

```bash
./occ talk:bot:install --help
```

Required arguments include the bot name, a webhook URL (your server's endpoint), and a shared secret. For internal Nextcloud applications, use the `nextcloudapp://$APPID` URI scheme with feature flag `4`.

## Authentication & Signature Verification

Every incoming webhook request includes a shared secret. Verification must happen before processing any payload:

### Required Headers

| Header | Description |
|---|---|
| `X-NEXTCLOUD-TALK-SIGNATURE` | 64-character hexadecimal HMAC-SHA256 digest |
| `X-NEXTCLOUD-TALK-RANDOM` | 64-character alphanumeric random nonce |
| `X-NEXTCLOUD-TALK-BACKEND` | Originating Nextcloud server URI |

### Verification Process

1. Concatenate the `X-NEXTCLOUD-TALK-RANDOM` nonce with the raw request body
2. Compute HMAC-SHA256 using the bot's shared secret as the key
3. Compare the result against `X-NEXTCLOUD-TALK-SIGNATURE` using a constant-time comparison (e.g., `hash_equals` in PHP)
4. Reject the request if signatures do not match

```python
import hmac
import hashlib

def verify_signature(body: bytes, random: str, signature: str, secret: str) -> bool:
    computed = hmac.new(secret.encode(), (random + body).encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)
```

**Critical: Always use constant-time comparison to prevent timing attacks.**

## Base Endpoint

All API calls route through:

```
POST /ocs/v2.php/apps/spreed/api/v1/bot/{token}/{action}
```

Where `{token}` is the conversation token and `{action}` is the specific operation.

**All outgoing requests must include:** `OCS-APIRequest: true`

## Incoming Webhooks (Events Received by the Bot)

All payloads use the **Activity Streams 2.0 Vocabulary** JSON format with `actor`, `object`, and optional `target` fields.

### Supported Event Types

#### 1. Chat Message (`Create`)

Triggered when a user sends a chat message in a room the bot is part of.

```json
{
  "type": "Create",
  "actor": {
    "id": "users/<userId>",
    "actorType": "users",
    "displayName": "<displayName>"
  },
  "actor.talkParticipantType": 1,
  "object": {
    "token": "<conversationToken>",
    "type": "comment",
    "id": "<messageId>",
    "content": "{\"message\":\"Hello bot!\",\"parameters\":{}}",
    "inReplyTo": "<parentMessageId or 0>"
  }
}
```

- `object.content` is a JSON-encoded dict with `message` (Markdown text) and `parameters` (rich objects)
- `object.inReplyTo` is present only for reply messages (message ID of the parent)
- Message length limit: ~32,000 characters

#### 2. Reaction Added (`Like`)

Triggered when a user adds an emoji reaction to any message.

```json
{
  "type": "Like",
  "actor": {
    "id": "users/<userId>",
    "actorType": "users"
  },
  "content": "<emoji>"
}
```

- `content` contains the emoji string (e.g., `:thumbsup:` or the raw emoji)
- Requires `reaction` capability flag (Talk 21+)

#### 3. Reaction Removed (`Undo`)

Triggered when a user removes an emoji reaction.

```json
{
  "type": "Undo",
  "actor": {
    "id": "users/<userId>",
    "actorType": "users"
  },
  "object": {
    "content": "<emoji>"
  }
}
```

- `object.content` contains the removed emoji
- Requires `reaction` capability flag (Talk 21+)

#### 4. Bot Added to Room (`Join`)

Triggered when the bot is added to a conversation.

```json
{
  "type": "Join",
  "actor": {
    "id": "bots/<sha1hash>",
    "actorType": "bots",
    "displayName": "<botName>"
  },
  "object": {
    "token": "<conversationToken>",
    "type": "room"
  }
}
```

- `actor.id` uses `bots/` prefix with a SHA1 hash
- `object.id` holds the conversation token

#### 5. Bot Removed from Room (`Leave`)

Triggered when the bot is removed from a conversation. Same structure as `Join` event.

## Outgoing Actions (Bot Sends to Chat)

### Send Text Message

```
POST /ocs/v2.php/apps/spreed/api/v1/bot/{token}/message
```

| Parameter | Description |
|---|---|
| `message` | The text message to send (Markdown supported) |
| `replyTo` | Message ID to reply to (optional) |
| `referenceId` | Unique reference ID for idempotency (optional) |
| `silent` | Send without notification (optional) |

**Response codes:**

| Code | Meaning |
|---|---|
| 201 | Created - message sent successfully |
| 400 | Bad Request - invalid reply reference or parameters |
| 401 | Unauthorized - signature verification failed |
| 404 | Not Found - conversation does not exist |
| 413 | Payload Too Large - exceeds message length limit |
| 429 | Too Many Requests - repeated auth failures |

### Add Reaction

```
POST /ocs/v2.php/apps/spreed/api/v1/bot/{token}/reaction/{messageId}
```

| Parameter | Description |
|---|---|
| `reaction` | Emoji string to add |

**Response codes:** 201 (success), 400 (invalid emoji), 401/404/429 (as above).

### Remove Reaction

```
DELETE /ocs/v2.php/apps/spreed/api/v1/bot/{token}/reaction/{messageId}
```

| Parameter | Description |
|---|---|
| `reaction` | Emoji string to remove |

**Response codes:** 200 (success), 401/404/429 (as above).

## Capabilities

| Capability | Description | Availability |
|---|---|---|
| `bots-v1` | Required for all basic bot operations | Always available |
| `reaction` | Enables emoji add/remove webhook events | Talk 21+ |
| `events` (flag 4) | Bypasses external webhooks; listens directly to internal PHP events | Talk 21+ |

### Internal Event Mode (`events` flag)

Instead of external webhooks, bots can hook directly into the Nextcloud PHP event system:

1. Configure bot URL using `nextcloudapp://$APPID` scheme
2. Set feature flag `4`
3. Reduce server load by eliminating external webhook roundtrips

## Error Handling

| HTTP Code | Meaning | Bot Action |
|---|---|---|
| 200 | OK | Standard successful response |
| 201 | Created | Message/reaction sent successfully |
| 400 | Bad Request | Check parameters - invalid message, reply reference, or emoji format |
| 401 | Unauthorized | Signature mismatch - verify shared secret and HMAC computation |
| 404 | Not Found | Invalid conversation token or message ID |
| 413 | Payload Too Large | Message exceeds ~32,000 character limit |
| 429 | Too Many Requests | Rate limited due to repeated auth failures - back off |

## Implementation Checklist for a Bot Channel

1. **HTTP server** - Expose an HTTPS endpoint for webhook deliveries
2. **Signature verification** - HMAC-SHA256 verification on every incoming request
3. **Event dispatch** - Route `Create`, `Like`, `Undo`, `Join`, `Leave` events to appropriate handlers
4. **Message sending** - Implement `POST /bot/{token}/message` calls with `OCS-APIRequest: true` header
5. **Reaction support** - Implement add/remove reaction endpoints if `reaction` capability is available
6. **Conversation tracking** - Track which rooms the bot has been `Join`ed to; ignore events from unknown rooms
7. **Idempotency** - Use `referenceId` parameter for message sends to handle retries
8. **Error handling** - Properly handle 4xx/5xx responses with retries/backoff where appropriate
9. **Message routing** - Map incoming messages to agent sessions; handle `inReplyTo` for conversation context