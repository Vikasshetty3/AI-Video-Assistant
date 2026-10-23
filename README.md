# 🎥 AI Video Assistant

An AI-powered application that transforms video and meeting content into searchable, actionable information using transcription, summarization, and Retrieval-Augmented Generation (RAG).

## 🚀 Overview

The AI Video Assistant helps users understand lengthy video or meeting content without manually reviewing the entire recording. It organizes the extracted information into summaries, action items, key decisions, and questions, with a conversational interface for asking questions about the content.

## ✨ Features

- **Transcription:** Convert supported audio/video content into text.
- **AI Summarization:** Generate concise summaries of the content.
- **Automatic Title Generation:** Create a meaningful title for the processed content.
- **Action Item Extraction:** Identify tasks and follow-up actions.
- **Key Decision Extraction:** Extract important decisions from conversations.
- **Question Extraction:** Identify relevant questions from the content.
- **RAG-Based Q&A:** Ask questions about the processed content and retrieve relevant context.
- **Interactive Interface:** Use a Streamlit application to interact with the results.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **LLM Integration:** Groq API, LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** Hugging Face Sentence Transformers
- **Text Processing:** LangChain text splitters
- **Environment Management:** uv

## 🏗️ Project Structure

```text
AI-Video-Assistant/
├── app.py
├── main.py
├── test.py
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── transcriber.py
│   └── vector_store.py
├── utils/
│   └── audio_processor.py
├── .env                 # Local secrets; never commit
├── .gitignore
├── pyproject.toml
├── requirement.txt
└── README.md
```

*The filenames shown above should match the actual files in your local project. Adjust the structure if you have renamed any modules.*

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vikasshetty3/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### 2. Install dependencies

Install [uv](https://docs.astral.sh/uv/) if it is not already installed.

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Replace the example key with your own API key from the Groq developer console. Never publish your actual API key.

### 4. Run the application

```bash
uv run streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 🔍 How It Works

1. **Input:** Provide supported video, audio, or meeting content.
2. **Processing:** Extract and transcribe the audio into text.
3. **Summarization:** Generate a concise summary and title.
4. **Information Extraction:** Identify action items, decisions, and questions.
5. **Indexing:** Split transcript text into chunks and represent them using embeddings in a ChromaDB vector store.
6. **Question Answering:** Retrieve relevant transcript chunks and use an LLM to generate context-aware answers.

## 🔐 Security

- Keep API keys in local environment variables.
- Do not commit `.env` files or other secrets.
- Keep virtual environments and generated data out of version control using `.gitignore`.

## 🎯 Use Cases

- Meeting and lecture summaries
- Video content analysis
- Extracting follow-up tasks from discussions
- Finding decisions and important details in long transcripts
- Asking questions about recorded content

## 🔮 Future Improvements

- Support additional video and meeting platforms.
- Improve transcription accuracy and processing speed.
- Add timestamps and citations to answers.
- Support persistent transcript libraries and multi-video search.
- Improve error handling and automated testing.

## 👨‍💻 Author

**Vikas S**

GitHub: [@Vikasshetty3](https://github.com/Vikasshetty3)

---

*Built as an AI application exploring transcription, LLM-powered summarization, information extraction, and Retrieval-Augmented Generation.*
