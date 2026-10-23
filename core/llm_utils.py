import os
import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_groq_llm(temperature: float = 0.3):
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
    If Groq returns a rate limit message specifying retry seconds, parses it and sleeps.
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
                # Parse retry time if available, e.g. "Please try again in 2.85s"
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

    raise last_err
