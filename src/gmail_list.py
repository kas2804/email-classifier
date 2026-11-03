# src/gmail_list.py
from __future__ import annotations
import base64
import http.server
import json
import os
import sys
import urllib.parse
import webbrowser
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TOKEN_PATH = os.path.join(PROJECT_ROOT, "token.json")
DEFAULT_CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "credentials.json")

def check_auth_status(token_path: str = DEFAULT_TOKEN_PATH) -> tuple[bool, str]:
    """
    Non-blocking check of whether token.json exists and is valid.
    """
    if not os.path.exists(token_path) and os.path.exists("token.json"):
        token_path = "token.json"
    if not os.path.exists(token_path):
        return False, "token.json not found."
    try:
        with open(token_path, "r") as f:
            data = json.load(f)
        if not data.get("refresh_token") and not data.get("token"):
            return False, "token.json is empty or invalid."
        return True, "Authenticated"
    except Exception as e:
        return False, f"Token error: {e}"

def run_lightweight_oauth(credentials_path: str, token_path: str, port: int = 8080) -> Credentials:
    """
    Lightweight, fast OAuth 2.0 flow using standard library http.server and requests.
    Eliminates heavy/slow oauthlib import chains that cause freezes on Mac.
    """
    import requests

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"Credentials file not found at '{credentials_path}'. "
            "Please ensure credentials.json is placed in your project root."
        )

    with open(credentials_path, "r") as f:
        data = json.load(f)
    client_info = data.get("installed") or data.get("web")
    if not client_info:
        raise ValueError("Invalid credentials.json: Missing 'installed' or 'web' client config.")

    client_id = client_info["client_id"]
    client_secret = client_info["client_secret"]
    redirect_uri = f"http://localhost:{port}"

    auth_params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(auth_params)

    auth_code = None

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if "code" in params:
                auth_code = params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"""
                <html>
                <body style="font-family: system-ui, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #2e7d32;">&#9989; Authentication Successful!</h1>
                    <p style="font-size: 18px;">Your token has been generated. You can close this tab and return to your app.</p>
                </body>
                </html>
                """)
            else:
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                err_msg = params.get("error", ["Unknown error"])[0]
                self.wfile.write(f"<h1>Authentication Failed</h1><p>{err_msg}</p>".encode("utf-8"))

        def log_message(self, format, *args):
            pass  # Suppress default noisy access logs

    print("\n" + "=" * 65)
    print("🔐 GMAIL AUTHORIZATION")
    print("=" * 65)
    print("\n👉 Click or open this link in your browser to authorize:")
    print(f"\n{auth_url}\n")
    print("=" * 65)
    print("Waiting for browser authorization...")

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    server = http.server.HTTPServer(("localhost", port), CallbackHandler)
    server.handle_request()
    server.server_close()

    if not auth_code:
        raise RuntimeError("Did not receive authorization code from Google.")

    print("Exchanging code for token...")
    token_url = "https://oauth2.googleapis.com/token"
    token_payload = {
        "code": auth_code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    resp = requests.post(token_url, data=token_payload, timeout=10)
    resp.raise_for_status()
    token_data = resp.json()

    saved_token = {
        "token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "token_uri": token_url,
        "client_id": client_id,
        "client_secret": client_secret,
        "scopes": SCOPES,
    }

    with open(token_path, "w") as f:
        json.dump(saved_token, f, indent=2)

    print(f"✅ Success! Token saved to: {token_path}")
    return Credentials.from_authorized_user_file(token_path, SCOPES)

def gmail_service(token_path: str = DEFAULT_TOKEN_PATH, credentials_path: str = DEFAULT_CREDENTIALS_PATH, open_browser: bool = True):
    """
    Initializes and returns the authorized Gmail API service resource.
    Uses fast local authentication and refreshes safely.
    """
    if not os.path.exists(token_path) and os.path.exists("token.json"):
        token_path = "token.json"
    if not os.path.exists(credentials_path) and os.path.exists("credentials.json"):
        credentials_path = "credentials.json"

    creds = None
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception as e:
            print(f"Could not read token ({e}). Re-authenticating...")
            creds = None

    if creds and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                import requests
                from google.auth.transport.requests import Request
                session = requests.Session()
                req = Request(session=session)
                creds.refresh(req)
            except Exception as e:
                print(f"Token refresh failed ({e}). Re-authenticating...")
                creds = None
        else:
            creds = None

    if not creds or not creds.valid:
        creds = run_lightweight_oauth(credentials_path, token_path)

    return build("gmail", "v1", credentials=creds)

def list_message_ids(max_n: int = 10, service=None):
    svc = service if service is not None else gmail_service()
    res = svc.users().messages().list(userId="me", maxResults=max_n).execute()
    return res.get("messages", [])

def get_message_brief(msg_id: str, service=None):
    svc = service if service is not None else gmail_service()
    msg = svc.users().messages().get(
        userId="me", id=msg_id, format="metadata",
        metadataHeaders=["Subject", "From"]
    ).execute()
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
    return {
        "id": msg_id,
        "subject": headers.get("subject", "(no subject)"),
        "from": headers.get("from", "(unknown)"),
        "snippet": msg.get("snippet", "")
    }

def batch_get_message_briefs(msg_ids: list[str], service=None) -> list[dict]:
    if not msg_ids:
        return []
    svc = service if service is not None else gmail_service()
    briefs = {}

    def callback(request_id, response, exception):
        if exception is None and response:
            headers = {h["name"].lower(): h["value"] for h in response.get("payload", {}).get("headers", [])}
            briefs[request_id] = {
                "id": request_id,
                "subject": headers.get("subject", "(no subject)"),
                "from": headers.get("from", "(unknown)"),
                "snippet": response.get("snippet", "")
            }
        else:
            briefs[request_id] = {
                "id": request_id,
                "subject": "(no subject)",
                "from": "(unknown)",
                "snippet": ""
            }

    try:
        batch = svc.new_batch_http_request(callback=callback)
        for mid in msg_ids:
            req = svc.users().messages().get(
                userId="me", id=mid, format="metadata",
                metadataHeaders=["Subject", "From"]
            )
            batch.add(req, request_id=mid)
        batch.execute()
        return [briefs[mid] for mid in msg_ids if mid in briefs]
    except Exception:
        return [get_message_brief(mid, service=svc) for mid in msg_ids]

def get_message_full(msg_id: str, service=None):
    svc = service if service is not None else gmail_service()
    msg = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
    subject = headers.get("subject", "(no subject)")

    def body_text(p):
        if p.get("body", {}).get("data"):
            try:
                return base64.urlsafe_b64decode(p["body"]["data"]).decode(errors="ignore")
            except Exception:
                return ""
        for part in p.get("parts", []) or []:
            t = body_text(part)
            if t:
                return t
        return ""

    body = body_text(msg.get("payload", {})) or msg.get("snippet", "")
    return {"id": msg_id, "subject": subject, "body": body}

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 GMAIL AUTHENTICATION TOOL (FAST NATIVE FLOW)")
    print("=" * 65)
    try:
        svc = gmail_service(open_browser=True)
        print("\nVerifying connection by fetching latest messages...")
        msgs = list_message_ids(max_n=5, service=svc)
        print(f"✅ Connection verified! Retrieved {len(msgs)} messages from inbox.")
        if msgs:
            mids = [m["id"] for m in msgs]
            samples = batch_get_message_briefs(mids, service=svc)
            print(f"Latest email: '{samples[0].get('subject')}' from {samples[0].get('from')}")
        print("\n🎉 Everything is ready! You can now run: streamlit run ui/app.py")
    except Exception as e:
        print(f"\n❌ Authentication error: {e}")
        sys.exit(1)
