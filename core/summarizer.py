import os
import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()


def get_llm(temperature: float = 0.3):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Check your .env file.")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=temperature,
        max_retries=3,
    )


def invoke_with_retry(runnable, input_data, max_retries: int = 5, default_wait: float = 3.5):
    """
    Invoke a LangChain runnable / chain / LLM with automatic 429 rate limit backoff.
    """
    wait_time = default_wait
    last_err = None
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                match = re.search(r"try again in ([0-9.]+)\s*s", err_str, re.IGNORECASE)
                if match:
                    parsed_wait = float(match.group(1)) + 0.6
                    sleep_sec = max(parsed_wait, 2.0)
                else:
                    sleep_sec = wait_time

                print(f"[Groq RateLimit 429] Waiting {sleep_sec:.2f}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_sec)
                wait_time = max(wait_time * 1.5, 3.5)
            else:
                raise e

    if last_err is not None:
        raise last_err
    raise RuntimeError("Invocation failed after maximum retries.")


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return ""

    llm = get_llm(temperature=0.3)

    # If the transcript is small enough, summarize directly in 1 pass
    if len(transcript) <= 4000:
        single_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert meeting summarizer. Provide a clear, "
                "professional summary using bullet points. Include key discussion "
                "points, decisions, action items, and deadlines when available. "
                "Do not invent information."
            ),
            ("human", "{text}"),
        ])
        chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | single_prompt
            | llm
            | StrOutputParser()
        )
        return invoke_with_retry(chain, transcript)

    # For longer transcripts, map-reduce
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize this portion of a meeting transcript "
            "concisely. Preserve important decisions, key points, "
            "action items, and deadlines."
        ),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)

    chunk_summaries = []
    for chunk in chunks:
        res = invoke_with_retry(map_chain, {"text": chunk})
        chunk_summaries.append(res)
        time.sleep(0.5)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine the "
            "partial summaries into one professional final summary "
            "using clear bullet points. Include key discussion "
            "points, decisions, action items, and deadlines when "
            "available. Do not invent information."
        ),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return invoke_with_retry(combined_chain, combined)


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    llm = get_llm(temperature=0.3)

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the meeting transcript, generate a short, "
            "professional title of no more than 8 words. "
            "Return only the title, without quotation marks "
            "or additional explanation."
        ),
        ("human", "{text}"),
    ])

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | title_prompt
        | llm
        | StrOutputParser()
    )

    return invoke_with_retry(title_chain, transcript[:1500]).strip()
