import os
import asyncio
import google.genai as genai
from google.genai import types


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
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Recipe.model_json_schema(),
        },
    )

    return response.text

async def generate_content_async(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-3-flash-preview",
) -> str:
    return await asyncio.to_thread(
        _generate_content_sync,
        client,
        system_prompt,
        user_prompt,
        model,
    )