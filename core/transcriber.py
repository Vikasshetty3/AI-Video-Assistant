import whisper
import os
import re
import time
import requests
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
if SARVAM_MODEL == "saaras:v2.5":
    SARVAM_MODEL = "saaras:v4"

_model = None


def load_model():

    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 


def clean_repetitions(text: str) -> str:
    """Detect and remove repetitive hallucination loops produced by Whisper."""
    if not text:
        return ""
    # Remove loops of 2-30 characters repeated 4+ times
    cleaned = re.sub(r'(.{2,30}?)\1{4,}', r'\1', text)
    # If the text is overwhelmingly composed of repeated characters (hallucination)
    unique_chars = set(cleaned.replace(" ", ""))
    if len(text) > 30 and len(unique_chars) <= 4:
        return ""
    return cleaned.strip()


def transcribe_chunk_whisper(chunk_path: str, language: str = None) -> str:
    if not os.path.exists(chunk_path):
        return ""

    model = load_model()

    kwargs = {
        "task": "transcribe",
        "temperature": (0.0, 0.2, 0.4),
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.2,
        "no_speech_threshold": 0.6,
        "fp16": False,
    }
    if language == "kannada":
        kwargs["language"] = "kn"
    elif language in ["hindi", "hinglish"]:
        kwargs["language"] = "hi"

    try:
        result = model.transcribe(chunk_path, **kwargs)
        raw_text = result.get("text", "")
        return clean_repetitions(raw_text)
    except Exception as e:
        print(f"Whisper primary transcription error ({e}). Trying robust fallback...")
        try:
            result = model.transcribe(
                chunk_path,
                task="transcribe",
                temperature=(0.0, 0.2),
                condition_on_previous_text=False,
                compression_ratio_threshold=2.2,
                fp16=False,
            )
            raw_text = result.get("text", "")
            return clean_repetitions(raw_text)
        except Exception as e2:
            print(f"Whisper fallback also failed ({e2}). Returning empty.")
            return ""


def _send_to_sarvam(
    piece_path: str,
    language_code: str
) -> str:
    """Send one ≤30s WAV file to Sarvam with automatic retry."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing. Check your .env file.")

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    last_err = None
    for attempt in range(3):
        try:
            with open(piece_path, "rb") as f:
                # Use clean ASCII filename in multipart form to avoid Windows encoding issues
                files = {
                    "file": ("audio_piece.wav", f, "audio/wav")
                }
                data = {
                    "model": SARVAM_MODEL,
                    "mode": "transcribe",
                    "language_code": language_code
                }

                response = requests.post(
                    SARVAM_STT_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=60
                )

            if response.status_code == 200:
                return response.json().get("transcript", "")
            elif response.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
            else:
                response.raise_for_status()
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    if last_err:
        raise last_err
    return ""


def transcribe_chunk_sarvam(
    chunk_path: str,
    language_code: str = "hi-IN"
) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    transcripts = []

    temp_dir = "scratch"
    os.makedirs(temp_dir, exist_ok=True)

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]
        # Use clean ASCII name in scratch/ to avoid Windows Unicode filename issues
        piece_path = os.path.join(temp_dir, f"sv_{os.getpid()}_{i}.wav")
        piece.export(piece_path, format="wav")

        try:
            text = _send_to_sarvam(
                piece_path,
                language_code
            )
            if text and text.strip():
                transcripts.append(text.strip())
        finally:
            if os.path.exists(piece_path):
                try:
                    os.remove(piece_path)
                except Exception:
                    pass

    return " ".join(transcripts).strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    lang = language.lower()

    if lang in ["hindi", "hinglish"]:
        try:
            return transcribe_chunk_sarvam(chunk_path, "hi-IN")
        except Exception as e:
            print(f"Sarvam Hindi STT error ({e}), falling back to Whisper...")
            return transcribe_chunk_whisper(chunk_path, language=lang)

    elif lang == "kannada":
        try:
            return transcribe_chunk_sarvam(chunk_path, "kn-IN")
        except Exception as e:
            print(f"Sarvam Kannada STT error ({e}), falling back to Whisper...")
            return transcribe_chunk_whisper(chunk_path, language="kannada")

    return transcribe_chunk_whisper(chunk_path)

def transcribe_all(chunks: list, language: str = "english") -> str:

    full_transcript = "" 

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)  

        if text and text.strip():
            full_transcript += text.strip() + " "  

    print("Transcription complete.")

    cleaned = full_transcript.strip()
    if not cleaned:
        return "No audible speech was detected in this recording."

    return cleaned  