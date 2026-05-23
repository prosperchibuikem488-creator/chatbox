import os
import sys
import time
import logging
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from model_loader import download_models
from app.inference.response_generator import ResponseGenerator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Solace Mental Health Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    logger.info("Downloading models if needed...")
    download_models()
    logger.info("Models ready.")

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

_sessions: dict = {}

def get_bot(session_id: str) -> ResponseGenerator:
    if session_id not in _sessions:
        logger.info(f"New session: {session_id}")
        _sessions[session_id] = ResponseGenerator()
    return _sessions[session_id]

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

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Solace chatbot API is running."}

@app.get("/health")
def detailed_health():
    return {
        "status":          "ok",
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
