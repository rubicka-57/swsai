from google import genai

from utils.config import get_settings


FALLBACK_ANSWER = "I don't have that information in the company documents."


def build_prompt(context: str, question: str) -> str:
    return f"""
You are a company document assistant.

Answer the user's question using ONLY the context below.
If the answer is not explicitly present in the context, respond exactly:
{FALLBACK_ANSWER}

Do not use outside knowledge. Do not guess.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def generate_grounded_answer(context: str, question: str) -> str:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=build_prompt(context, question),
    )
    answer = (response.text or "").strip()
    return answer or FALLBACK_ANSWER
