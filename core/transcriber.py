import whisper
import os
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


def transcribe_chunk_whisper(chunk_path: str, language: str = None) -> str:
    if not os.path.exists(chunk_path):
        return ""

    model = load_model()  

    kwargs = {
        "task": "transcribe",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "fp16": False,
    }
    if language == "kannada":
        kwargs["language"] = "kn"
    elif language in ["hindi", "hinglish"]:
        kwargs["language"] = "hi"

    try:
        result = model.transcribe(chunk_path, **kwargs)
        return result.get("text", "")
    except Exception as e:
        print(f"Whisper primary transcription error ({e}). Trying robust fallback...")
        try:
            # Fallback pass without language constraint
            result = model.transcribe(
                chunk_path,
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=False,
                fp16=False,
            )
            return result.get("text", "")
        except Exception as e2:
            print(f"Whisper fallback also failed ({e2}). Returning empty.")
            return ""  


def _send_to_sarvam(
    piece_path: str,
    language_code: str
) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:
        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
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
            timeout=120
        )

    response.raise_for_status()
    return response.json().get("transcript", "")


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

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            text = _send_to_sarvam(
                piece_path,
                language_code
            )
            transcripts.append(text)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

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