import os
import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

_extraction_cache = {}


def get_llm(temperature: float = 0.2):
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

    llm = get_llm(temperature=0.2)
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
