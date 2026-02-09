import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SPEAKER_WAV = "models/reference_audio.mp3"


class AudiobookGenerator:

    def __init__(self, model_name: str = DEFAULT_MODEL, default_speaker_wav: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.default_speaker_wav = default_speaker_wav or DEFAULT_SPEAKER_WAV

        if self.default_speaker_wav and not Path(self.default_speaker_wav).exists():
            logger.warning(f"Default speaker wav not found: {self.default_speaker_wav}. Falling back to no reference.")
            self.default_speaker_wav = None

        try:
            from TTS.api import TTS
            self.model = TTS(model_name, gpu=(self.device == "cuda"))
            logger.info(f"Coqui TTS model loaded on {self.device}")
        except Exception as e:
            logger.warning(f"TTS init failed: {e}. TTS will be disabled.")

    def is_available(self) -> bool:
        return self.model is not None

    def generate(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> str:
        if not self.is_available():
            raise RuntimeError("TTS model is not available.")

        ref = speaker_wav or self.default_speaker_wav

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if ref:
            self.model.tts_to_file(
                text=text,
                speaker_wav=ref,
                language=language,
                file_path=str(out),
            )
        else:
            self.model.tts_to_file(
                text=text,
                language=language,
                file_path=str(out),
            )

        logger.info(f"Audio saved to {out}")
        return str(out)
