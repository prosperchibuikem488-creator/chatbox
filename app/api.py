from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.inference.response_generator import ResponseGenerator
from app.utility.download_model import download_model

app = FastAPI()
bot = None

class ChatRequest(BaseModel):
    message: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ FIXED startup
@app.on_event("startup")
def startup_event():
    global bot

    print("🔄 Downloading model...")
    download_model()

    print("🧠 Loading model...")
    bot = ResponseGenerator()

    print("✅ Model ready!")


# ✅ Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        print("INPUT:", req.message)

        result = bot.generate(req.message)

        print("OUTPUT:", result)

        return {
            "debug": True,
            "result": str(result)
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "error": str(e)
        }

# ✅ Health check
@app.get("/")
def root():
    return {"status": "API is running 🚀"}