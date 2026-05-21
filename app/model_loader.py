"""
model_loader.py
Downloads models from Hugging Face on first startup.
Set HF_REPO_ID and HF_TOKEN in Render environment variables.
"""
import os
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

def download_models():
    repo_id = os.getenv("HF_REPO_ID", "chibuikem432/chatbox-5")
    token   = os.getenv("HF_TOKEN")

    # Model folders your code expects
    emotion_path = "models/emotion_classifier_v6"
    intent_path  = "models/intent_classifier_v4"

    # Skip download if models already exist (Render persistent disk)
    if os.path.exists(emotion_path) and os.path.exists(intent_path):
        logger.info("Models already exist locally — skipping download.")
        return

    logger.info(f"Downloading models from Hugging Face: {repo_id}")

    try:
        snapshot_download(
             repo_id    = repo_id,
             local_dir  = "models",
             token      = token,
             repo_type  = "model",
        )
        logger.info("Models downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        raise RuntimeError(
            f"Could not download models from {repo_id}. "
            "Check HF_REPO_ID and HF_TOKEN on Render."
        ) from e
