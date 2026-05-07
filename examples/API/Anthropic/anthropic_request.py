"""
Test script to request tool use from Anthropic endpoint.
"""
import asyncio
import httpx
import sys

# The LLM host - based on examples, likely 192.168.255.10:8123 or similar
BASE_URL = "http://192.168.255.10:8123"
MODEL = "qwen/qwen3.5-35b-a3b"

# Anthropic-compatible endpoint
CHAT_ENDPOINT = "/v1/messages"

# Tool definition
TOOLS = [{
    "name": "calculate",
    "description": "Calculate a mathematical expression",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}]

# Request body
body = {
    "model": MODEL,
    "stream": True,
    "max_tokens": 4096,
    "tools": TOOLS,
    "tool_choice": {"type": "any"},
    "messages": [
        {"role": "user", "content": "Calculate 2**16 + 32 * 15 - 100"}
    ]
}

async def capture_stream():
    print(f"Sending request to {BASE_URL}{CHAT_ENDPOINT}...")

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", f"{BASE_URL}{CHAT_ENDPOINT}", json=body) as response:
            print(f"Response status: {response.status_code}")
            lines = []
            async for line in response.aiter_lines():
                if line:
                    lines.append(line)
                    print(line)
    return lines

if __name__ == "__main__":
    lines = asyncio.run(capture_stream())

    # Save to file
    output_path = "/home/frygge/projects/private/peteos/examples/API/Anthropic/raw_sse_stream.txt"
    with open(output_path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"\nSaved {len(lines)} lines to {output_path}")