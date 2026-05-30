import os
import logging
from huggingface_hub import snapshot_download
 
logger = logging.getLogger(__name__)
 
MODEL_DIR = os.getenv("MODEL_DIR", "models")
 
def download_models():
    repo_id = os.getenv("HF_REPO_ID", "chibuikem432/chatbox-5")
    token   = os.getenv("HF_TOKEN")
 
    # Set cache dirs BEFORE any model loading to skip migration
    cache_dir = os.path.join(MODEL_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"]                    = cache_dir
    os.environ["TRANSFORMERS_CACHE"]         = cache_dir
    os.environ["HF_DATASETS_CACHE"]          = cache_dir
    os.environ["TRANSFORMERS_VERBOSITY"]     = "error"
 
    emotion_path = os.path.join(MODEL_DIR, "emotion_classifier_v6")
    intent_path  = os.path.join(MODEL_DIR, "intent_classifier_v4")
 
    emotion_done = os.path.exists(os.path.join(emotion_path, "model.safetensors"))
    intent_done  = os.path.exists(os.path.join(intent_path,  "model.safetensors"))
 
    if emotion_done and intent_done:
        logger.info("Models already exist — skipping download.")
        os.environ["EMOTION_MODEL_PATH"] = emotion_path
        os.environ["INTENT_MODEL_PATH"]  = intent_path
        return
 
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Downloading models from: {repo_id}")
 
    snapshot_download(
        repo_id   = repo_id,
        local_dir = MODEL_DIR,
        token     = token,
        repo_type = "model",
    )
 
    os.environ["EMOTION_MODEL_PATH"] = emotion_path
    os.environ["INTENT_MODEL_PATH"]  = intent_path
    logger.info("Models downloaded successfully.")
