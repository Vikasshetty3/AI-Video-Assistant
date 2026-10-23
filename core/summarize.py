import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm_utils import get_groq_llm, invoke_with_retry

load_dotenv()


def get_llm():
    return get_groq_llm(temperature=0.3)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return ""

    llm = get_llm()

    # If the transcript is small enough, summarize directly in 1 pass to save tokens & avoid 429
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

    llm = get_llm()

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

    # Limit to first 1500 chars to avoid exceeding TPM rate limits
    return invoke_with_retry(title_chain, transcript[:1500]).strip()
