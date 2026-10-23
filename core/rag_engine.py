import os
import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

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


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = invoke_with_retry(rag_chain, question)
    print(f"Answer: {answer}")
    return answer