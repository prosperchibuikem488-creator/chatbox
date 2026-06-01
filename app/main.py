import os
# Set cache dirs first to prevent slow migration
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"

import sys
import time
import logging
import asyncio
import threading
import requests as http_requests
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Solace Mental Health Chatbot API", version="1.0.0")

# ✅ FIXED CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Keep-alive ping ────────────────────────────────────────
async def keep_alive():
    url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not url:
        return
    while True:
        await asyncio.sleep(600)
        try:
            http_requests.get(f"{url}/", timeout=10)
            logger.info("Keep-alive ping sent.")
        except Exception as e:
            logger.warning(f"Keep-alive ping failed: {e}")

# ── Global bot instance preloaded at startup ────────────────
_preloaded_bot = None

def _preload_in_background():
    global _preloaded_bot
    from model_loader import download_models
    from inference.response_generator import ResponseGenerator
    download_models()
    _preloaded_bot = ResponseGenerator()
    logger.info("Models preloaded and ready.")

@app.on_event("startup")
async def startup():
    asyncio.create_task(keep_alive())
    # Load models in background thread — doesn't block startup
    thread = threading.Thread(target=_preload_in_background, daemon=True)
    thread.start()
    logger.info("Solace API is ready.")

# ── Rate limiting ──────────────────────────────────────────
RATE_LIMIT  = 20
RATE_WINDOW = 60
_req_log: dict = defaultdict(list)

def check_rate_limit(ip: str):
    now = time.time()
    _req_log[ip] = [t for t in _req_log[ip] if now - t < RATE_WINDOW]
    if len(_req_log[ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down.",
        )
    _req_log[ip].append(now)

# ── Session store ──────────────────────────────────────────
_sessions: dict = {}

def get_bot(session_id: str):
    from inference.response_generator import ResponseGenerator
    if session_id not in _sessions:
        if _preloaded_bot is not None:
            _sessions[session_id] = _preloaded_bot
        else:
            _sessions[session_id] = ResponseGenerator()
        logger.info(f"New session: {session_id}")
    return _sessions[session_id]

# ── Models ─────────────────────���───────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response:           str
    emotion:            str
    secondary_emotions: list[str]
    intent:             str

class ResetRequest(BaseModel):
    session_id: str = "default"

# ── Routes ─────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Solace chatbot API is running."}

@app.get("/health")
def detailed_health():
    return {
        "status":          "ok",
        "models_ready":    _preloaded_bot is not None,
        "active_sessions": len(_sessions),
        "timestamp":       datetime.utcnow().isoformat(),
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    check_rate_limit(req.client.host)
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(request.message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 chars).")

    logger.info(f"[{request.session_id}] User: {request.message[:80]}")
    try:
        bot    = get_bot(request.session_id)
        result = bot.generate(request.message.strip())
        logger.info(f"[{request.session_id}] Emotion: {result['emotion']} | Intent: {result['intent']}")
        return ChatResponse(
            response           = result["response"],
            emotion            = result["emotion"],
            secondary_emotions = result.get("secondary_emotions", []),
            intent             = result["intent"],
        )
    except Exception as e:
        logger.error(f"[{request.session_id}] Error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")

@app.post("/reset")
async def reset(request: ResetRequest, req: Request):
    check_rate_limit(req.client.host)
    sid = request.session_id
    if sid in _sessions:
        bot = _sessions[sid]
        bot.dialog_model.chat_history = []
        bot._reset_session()
        logger.info(f"Session reset: {sid}")
        return {"status": "reset", "session_id": sid}
    return {"status": "no_session", "session_id": sid}
