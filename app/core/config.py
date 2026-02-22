from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "AuraLearn"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    DESCRIPTION: str = """AuraLearn API provides comprehensive document processing capabilities:
    - **PDF Ingestion**: Upload and process PDF documents with intelligent chunking
    - **Extractive Summarization**: Extract key sentences using BiLSTM neural model
    - **Abstractive Summarization**: Generate concise summaries using fine-tuned T5
    - **Audiobook Generation**: Convert text to natural-sounding speech with TTS
    - **Complete Pipeline**: End-to-end processing from PDF to audiobook
    """
    
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    DATA_DIR: Path = BASE_DIR / "data"
    
    EXTRACTIVE_MODEL_PATH: str = "models/extractive_model_final.pt"
    ABSTRACTIVE_MODEL_PATH: str = "models/t5_summarizer"
    
    SENTENCE_ENCODER: str = "all-MiniLM-L6-v2"
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    IMAGE_CAPTION_MODEL: str = "Salesforce/blip-image-captioning-base"
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    
    MAX_PDF_SIZE: int = 50 * 1024 * 1024
    MAX_AUDIO_SIZE: int = 10 * 1024 * 1024
    
    DEFAULT_EXTRACTIVE_SENTENCES: int = 5
    DEFAULT_ABSTRACTIVE_MAX_LENGTH: int = 150
    DEFAULT_ABSTRACTIVE_MIN_LENGTH: int = 40
    
    ALLOWED_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)