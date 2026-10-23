import os
import sys
import glob
import yt_dlp
from pydub import AudioSegment

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_template = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(raw_filename)
        expected_wav = base + ".wav"

        if os.path.exists(expected_wav):
            return expected_wav

        # Fallback to any freshly created wav file in downloads
        wav_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.wav"))
        if wav_files:
            return max(wav_files, key=os.path.getmtime)

        return expected_wav


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16kHz mono WAV format using pydub."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found: {input_path}")

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16kHz mono
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found for chunking: {wav_path}")

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    # If shorter than chunk_minutes, export a single chunk
    if len(audio) <= chunk_ms:
        chunk_path = f"{wav_path}_chunk_0.wav"
        audio.export(chunk_path, format="wav")
        return [chunk_path]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    source = source.strip().strip('"').strip("'")
    if not source:
        raise ValueError("Source path or URL cannot be empty.")

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        raw_wav = download_youtube_audio(source)
        # Standardize to 16kHz mono for optimal STT accuracy
        print("Standardizing audio to 16kHz mono WAV...")
        wav_path = convert_to_wav(raw_wav)
    else:
        print("Detected local file. Converting to 16kHz mono WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
