
import os
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JARVIS")

app = FastAPI(title="JARVIS AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are JARVIS, an advanced personal AI assistant.

PERSONALITY:
- Intelligent, composed and sophisticated.
- Calm, articulate and quietly witty.
- Warm, helpful and professional.
- Observant and thoughtful.

BEHAVIOUR:
- Understand the user's intent.
- Explain complicated things clearly.
- Give step-by-step instructions when needed.
- Speak naturally, not like a robot.
- Never invent facts or claim abilities you do not have.

IDENTITY:
You are JARVIS.
You have no human face or physical body.
Your visual identity is a futuristic blue-cyan holographic Arc Reactor.
"""

class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=30)


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured."
        )

    retry_options = types.HttpRetryOptions(
        attempts=3,
        initial_delay=1.0,
        max_delay=8.0,
        exp_base=2.0,
        jitter=1.0,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=retry_options
        ),
    )


def make_prompt(messages):
    if messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from the user."
        )

    if any(m.role not in ["user", "assistant"] for m in messages):
        raise HTTPException(
            status_code=400,
            detail="Invalid message role."
        )

    conversation = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in messages[-12:]
    )

    return SYSTEM_PROMPT + "\n\nConversation:\n" + conversation + "\n\nJARVIS:"


@app.get("/")
def home():
    return {"status": "JARVIS online", "ai": "Gemini"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# Original endpoint remains available.
@app.post("/api/chat")
def chat(data: ChatRequest):
    client = get_client()
    prompt = make_prompt(data.messages)
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=700
            )
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        return {"reply": response.text}

    except HTTPException:
        raise
    except Exception:
        logger.exception("Gemini request failed.")
        raise HTTPException(
            status_code=502,
            detail="JARVIS Gemini service encountered an error."
        )


# New streaming endpoint.
@app.post("/api/chat/stream")
def chat_stream(data: ChatRequest):
    client = get_client()
    prompt = make_prompt(data.messages)
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    def generate():
        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=700
                )
            )

            for chunk in stream:
                if chunk.text:
                    payload = json.dumps({"text": chunk.text})
                    yield f"data: {payload}\n\n"

            yield "data: [DONE]\n\n"

        except Exception:
            logger.exception("Gemini streaming failed.")
            payload = json.dumps({
                "error": "JARVIS encountered a generation error."
            })
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
