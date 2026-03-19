#!/usr/bin/env python3
"""MCP server that sends WhatsApp alerts via Twilio."""

import json
import os
import sys
from twilio.rest import Client


def send_whatsapp(to: str, body: str) -> str:
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM")

    if not all([account_sid, auth_token, from_number]):
        return json.dumps({
            "error": "Missing Twilio credentials. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_FROM env vars."
        })

    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=body,
        from_=f"whatsapp:{from_number}",
        to=f"whatsapp:{to}"
    )
    return json.dumps({"success": True, "sid": message.sid, "status": message.status})


TOOLS = [
    {
        "name": "send_alert",
        "description": "Send a WhatsApp message to alert the user about training run issues, errors, or status updates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message body to send via WhatsApp."
                }
            },
            "required": ["message"]
        }
    }
]


def handle_request(request):
    method = request.get("method")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "sms-alert", "version": "1.0.0"}
        }
    elif method == "tools/list":
        return {"tools": TOOLS}
    elif method == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"].get("arguments", {})
        if tool_name == "send_alert":
            to = os.environ.get("WHATSAPP_TO_NUMBER")
            if not to:
                return {
                    "content": [{"type": "text", "text": "Missing WHATSAPP_TO_NUMBER env var."}],
                    "isError": True
                }
            result = send_whatsapp(to, args["message"])
            return {
                "content": [{"type": "text", "text": result}]
            }
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            "isError": True
        }
    elif method == "notifications/initialized":
        return None
    else:
        return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main():
    """Run MCP server over stdio using JSON-RPC."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = request.get("id")
        result = handle_request(request)

        if result is None:
            continue

        response = {"jsonrpc": "2.0", "id": req_id}
        if "error" in result:
            response["error"] = result["error"]
        else:
            response["result"] = result

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
