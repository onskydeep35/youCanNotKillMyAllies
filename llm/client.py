import os
import google.genai as genai
import asyncio


def create_gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    return genai.Client(api_key=api_key)


def _generate_content_sync(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=[
            {"role": "system", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
    )
    return response.text or ""


async def generate_content_async(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-3-flash-preview",
) -> str:
    """
    Run blocking Gemini call in a background thread.
    """
    return await asyncio.to_thread(
        _generate_content_sync,
        client,
        system_prompt,
        user_prompt,
        model,
    )
