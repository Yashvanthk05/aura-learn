import whisper
import ffmpeg
from pathlib import Path
from typing import Dict, Union
from app.core.config import settings

class TranscriptionService:
    def __init__(self):
        # Check for ffmpeg
        import shutil
        if not shutil.which("ffmpeg"):
            print("Error: ffmpeg is not installed or not in PATH.")
            raise RuntimeError("ffmpeg is required but not found. Please install it (e.g., 'sudo apt install ffmpeg').")

        print(f"Loading Whisper model: {settings.WHISPER_MODEL_SIZE} on {settings.WHISPER_DEVICE}...")
        try:
            self.model = whisper.load_model(settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise e

    def transcribe(self, file_path: Union[str, Path]) -> Dict:
        """
        Transcribes an audio or video file using OpenAI Whisper.
        
        Args:
            file_path: Path to the audio/video file.
            
        Returns:
            Dictionary containing the transcription text and segments.
        """
        file_path = str(file_path)
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            result = self.model.transcribe(file_path)
            
            return {
                "text": result["text"].strip(),
                "segments": result["segments"],
                "language": result["language"]
            }
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
