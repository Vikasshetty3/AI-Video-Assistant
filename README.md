# 🎬 AI Video Assistant

An AI-powered video and meeting intelligence application that transforms YouTube videos and audio/video recordings into searchable, actionable information using speech recognition, LLM-powered summarization, information extraction, and Retrieval-Augmented Generation (RAG).

## 🚀 Overview

AI Video Assistant helps users understand long videos, lectures, and meeting recordings without manually reviewing the entire content.

The application processes media, transcribes spoken content, generates a title and summary, extracts action items and key decisions, identifies open questions, and provides an interactive chat interface for asking questions about the transcript.

Built with Python, Streamlit, LangChain, Groq, Whisper, Sarvam AI, and ChromaDB, the project combines speech-to-text processing with LLM-based analysis and semantic retrieval.

## ✨ Features

- **YouTube URL Processing:** Enter a YouTube video URL to retrieve and process its audio.
- **Direct Media Upload:** Upload supported audio or video files without downloading them through YouTube.
- **Multilingual Transcription:** Supports English, Hindi, Hinglish, and Kannada through the configured transcription pipeline.
- **Speech Recognition:** Uses OpenAI Whisper for the English transcription path and Sarvam AI for configured Indic-language transcription paths.
- **Automatic Title Generation:** Generate a concise, meaningful title based on the transcript.
- **AI Summarization:** Generate summaries of long recordings using map-reduce summarization.
- **Action Item Extraction:** Identify tasks, responsibilities, and deadlines when mentioned in the conversation.
- **Key Decision Extraction:** Extract important decisions made during a meeting or discussion.
- **Open Question Extraction:** Identify questions and topics that may require further discussion.
- **RAG-Based Question Answering:** Ask questions about the processed content and generate answers using relevant transcript passages.
- **Semantic Search:** Generate text embeddings and retrieve relevant transcript chunks using ChromaDB.
- **Interactive Streamlit UI:** View results, track pipeline progress, and chat with the processed content.
- **Command-Line Interface:** Run the processing pipeline and ask questions through the terminal using `main.py`.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application and pipeline logic |
| Streamlit | Interactive web interface |
| yt-dlp | YouTube audio acquisition |
| FFmpeg | Media conversion and audio extraction |
| Pydub | Audio processing and chunking |
| OpenAI Whisper | Local speech recognition |
| Sarvam AI | API-based speech recognition for configured Indic languages |
| Groq API | Large language model inference |
| LangChain | Prompt templates, model orchestration, and RAG pipeline |
| Hugging Face Sentence Transformers | Semantic text embeddings |
| ChromaDB | Vector storage and similarity search |
| LangChain Text Splitters | Transcript chunking |
| python-dotenv | Environment variable configuration |
| uv | Python dependency and environment management |

## 🏗️ Project Structure

```text
AI-Video-Assistant/
│
├── app.py
│   └── Streamlit UI and application pipeline
│
├── main.py
│   └── Command-line application entry point
│
├── core/
│   ├── transcriber.py
│   │   └── Speech-to-text processing
│   │
│   ├── summarize.py
│   │   └── Title generation and summarization
│   │
│   ├── extractor.py
│   │   └── Action items, decisions, and open questions
│   │
│   ├── RAG_engine.py
│   │   └── Retrieval-augmented question answering
│   │
│   └── vector_store.py
│       └── Embeddings, ChromaDB, and retrieval
│
├── utils/
│   └── audio_processing.py
│       └── YouTube downloads, media conversion, and audio chunking
│
├── .env.example
│   └── Environment variable template
│
├── .gitignore
│
├── pyproject.toml
│   └── Project metadata and dependency declarations
│
├── uv.lock
│   └── Locked dependency versions
│
├── requirement.txt
│   └── Alternative dependency specification
│
└── README.md
```

*Keep this tree synchronized with the files actually committed to your repository.*

## ⚙️ Getting Started

### Prerequisites

- Python version compatible with the project's dependencies (Python 3.10 or newer is recommended).
- [uv](https://docs.astral.sh/uv/) for dependency management.
- FFmpeg and FFprobe installed and accessible from your system's PATH.
- A Groq API key.
- A Sarvam AI API key for the configured Sarvam transcription paths.
- Internet access for YouTube downloads and external AI APIs.

### 1. Clone the repository

```bash
git clone https://github.com/Vikasshetty3/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### 2. Install dependencies

Install `uv` if you have not already installed it.

```bash
uv sync
```

Ensure FFmpeg is installed separately. The Python package installation does not necessarily install the FFmpeg executable.

### 3. Configure environment variables

Create a `.env` file in the project root. You can use `.env.example` as a template.

```env
# Groq LLM configuration
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

# Whisper configuration
WHISPER_MODEL=small

# Sarvam AI speech-to-text configuration
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v4
```

Replace the placeholders with your own credentials and use model identifiers supported by your account and installed code.

**Security:** Never commit your actual `.env` file, API keys, browser cookies, or other authentication credentials to GitHub.

### 4. Run the Streamlit application

```bash
uv run streamlit run app.py
```

Open the local address shown in the terminal. The default Streamlit address is usually:

```text
http://localhost:8501
```

### 5. Run the command-line application

```bash
uv run python main.py
```

Follow the terminal prompts to enter a media source and interact with the pipeline.

## ▶️ How to Use the Application

1. Launch the Streamlit application.
2. Enter a YouTube video URL or provide a supported local media file path.
3. Alternatively, use the **Upload media** option in the sidebar to upload an audio or video file.
4. Select the desired language: English, Hindi, Hinglish, or Kannada.
5. Click **Analyse**.
6. Wait for media processing, transcription, title generation, summarization, information extraction, and RAG initialization to complete.
7. Review the generated title, summary, transcript, action items, key decisions, and open questions.
8. Use the chat interface to ask questions about the processed recording.

Choose one input method at a time: enter a URL/path or upload a file.

### Supported Upload Formats

The Streamlit interface accepts the following file extensions:

`MP4`, `MOV`, `MKV`, `AVI`, `WEBM`, `MP3`, `WAV`, `M4A`, `MPEG`, `MPGA`, `OGG`, and `FLAC`.

Actual media compatibility depends on the installed FFmpeg build and the input file's encoding.

### YouTube Download Notes

YouTube downloads depend on video availability, network conditions, and YouTube's access restrictions. Some videos may require authentication or may not be downloadable through `yt-dlp`.

For videos that cannot be downloaded, use a media file you are authorized to access, or try again later. Browser authentication, if configured locally, must remain private and must never be committed to the repository.

## 🧠 How It Works

The application follows a modular AI pipeline.

### 1. Media Ingestion

The media-processing module accepts a YouTube URL or a local audio/video file. It downloads or converts the media and prepares WAV audio chunks for transcription.

### 2. Speech-to-Text Transcription

The transcription module processes audio chunks using the configured transcription backend. Whisper handles the English path, while Sarvam AI handles the configured Indic-language paths.

The resulting text is combined into a transcript for downstream analysis.

### 3. Title Generation and Summarization

The transcript is passed to LLM-powered functions that generate a title and a concise summary.

For long transcripts, the summarization pipeline divides the text into chunks, summarizes the individual chunks, and combines the partial summaries into a final summary.

### 4. Information Extraction

Separate prompt-driven functions extract:

- Action items and follow-up tasks
- Key decisions
- Open questions

The extracted results are displayed in the Streamlit interface.

### 5. Vector Indexing

The transcript is split into smaller text chunks. A sentence-transformer embedding model converts the chunks into numerical vectors, which are stored in ChromaDB for semantic retrieval.

### 6. Retrieval-Augmented Generation

When a user asks a question, the retriever searches for relevant transcript chunks. The retrieved context and question are passed to the configured LLM, which generates an answer based on the supplied content.

This allows users to ask targeted questions without manually searching through the full transcript.

## 🧩 Key Technical Concepts

**Speech Recognition:** Converts spoken language into text using pretrained speech-recognition models and APIs.

**Map-Reduce Summarization:** Summarizes individual text chunks and combines their summaries to handle longer transcripts.

**Embeddings:** Represent text as numerical vectors that support semantic similarity search.

**Vector Database:** ChromaDB stores transcript chunks and their embeddings for retrieval.

**Retrieval-Augmented Generation:** Retrieves relevant transcript content and supplies it to an LLM as context when answering questions.

**Modular Architecture:** Separates media processing, transcription, summarization, extraction, retrieval, and UI logic into distinct modules.

## 🔐 Security and Privacy

- Store API keys in local environment variables.
- Keep `.env` out of version control.
- Never commit browser cookies or exported authentication files.
- Do not upload credentials or private meeting recordings to public repositories.
- Review `.gitignore` before committing generated files, local databases, virtual environments, and temporary media.
- Remember that audio sent to external transcription APIs is processed according to the provider's policies.

## ⚠️ Limitations

- YouTube download availability depends on YouTube's restrictions and `yt-dlp` compatibility.
- Transcription quality depends on audio clarity, language, accents, and background noise.
- Generated summaries and extracted items may omit details or contain errors.
- RAG answers depend on the quality of the transcript and the relevance of retrieved chunks.
- The application does not independently verify extracted task assignments or decisions.
- Processing speed depends on recording length, local hardware, model configuration, and external API latency.
- External API features require valid credentials and network access.

## 🎯 Use Cases

- Summarizing lectures and educational videos
- Meeting minutes and discussion summaries
- Extracting tasks and deadlines from conversations
- Identifying key decisions and unresolved questions
- Searching long recordings through natural-language questions
- Reviewing technical presentations and tutorials
- Building searchable notes from recorded content

## 🔮 Future Improvements

- Add timestamp-aware answers and source citations.
- Support persistent libraries of multiple recordings.
- Add speaker identification and diarization.
- Improve retrieval quality and evaluate answer faithfulness.
- Add automated tests and more robust error handling.
- Add export options for transcripts, summaries, and action items.
- Support additional transcription languages and media sources.
- Improve caching, performance, and processing observability.

## 👨‍💻 Author

**Vikas S**

- GitHub: [@Vikasshetty3](https://github.com/Vikasshetty3)
- Repository: [AI-Video-Assistant](https://github.com/Vikasshetty3/AI-Video-Assistant)

---

*Built to explore speech recognition, LLM-powered summarization, structured information extraction, semantic search, and Retrieval-Augmented Generation.*
