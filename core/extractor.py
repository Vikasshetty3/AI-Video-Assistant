import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_utils import get_groq_llm, invoke_with_retry

_extraction_cache = {}


def get_llm():
    return get_groq_llm(temperature=0.2)


def extract_all_items(transcript: str) -> dict:
    if not transcript or not transcript.strip():
        return {
            "action_items": "No action items found.",
            "key_decisions": "No key decisions found.",
            "open_questions": "No open questions found.",
        }

    # Cache lookup to avoid duplicate Groq calls within the same run
    cache_key = hash(transcript.strip())
    if cache_key in _extraction_cache:
        return _extraction_cache[cache_key]

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting analyst. Analyze the provided meeting transcript and extract three distinct sections.
Return your response using EXACTLY these markdown headers so each section can be extracted:

### ACTION ITEMS
List all action items with:
- Task description
- Owner (who is responsible)
- Deadline (if mentioned, else write 'Not specified')
(If none found, write 'No action items found.')

### KEY DECISIONS
List all key decisions made as a numbered list.
(If none found, write 'No key decisions found.')

### OPEN QUESTIONS
List all unresolved questions or topics needing follow-up as a numbered list.
(If none found, write 'No open questions found.')"""
        ),
        ("human", "{text}"),
    ])

    chain = prompt | llm | StrOutputParser()
    raw_response = invoke_with_retry(chain, {"text": transcript})

    action_items = "No action items found."
    key_decisions = "No key decisions found."
    open_questions = "No open questions found."

    ai_match = re.search(r"###\s*ACTION ITEMS\s*\n(.*?)(?=###|\Z)", raw_response, re.DOTALL | re.IGNORECASE)
    kd_match = re.search(r"###\s*KEY DECISIONS\s*\n(.*?)(?=###|\Z)", raw_response, re.DOTALL | re.IGNORECASE)
    oq_match = re.search(r"###\s*OPEN QUESTIONS\s*\n(.*?)(?=###|\Z)", raw_response, re.DOTALL | re.IGNORECASE)

    if ai_match and ai_match.group(1).strip():
        action_items = ai_match.group(1).strip()
    if kd_match and kd_match.group(1).strip():
        key_decisions = kd_match.group(1).strip()
    if oq_match and oq_match.group(1).strip():
        open_questions = oq_match.group(1).strip()

    result = {
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
    }
    _extraction_cache[cache_key] = result
    return result


def extract_action_items(transcript: str) -> str:
    return extract_all_items(transcript)["action_items"]


def extract_key_decisions(transcript: str) -> str:
    return extract_all_items(transcript)["key_decisions"]


def extract_questions(transcript: str) -> str:
    return extract_all_items(transcript)["open_questions"]
