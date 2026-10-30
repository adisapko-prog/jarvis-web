
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


@app.get("/")
def home():
    return {"status": "JARVIS online", "ai": "Gemini"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/chat")
def chat(data: ChatRequest):

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured."
        )

    if data.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from the user."
        )

    if any(m.role not in ["user", "assistant"] for m in data.messages):
        raise HTTPException(
            status_code=400,
            detail="Invalid message role."
        )

    conversation = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in data.messages[-12:]
    )

    prompt = (
        SYSTEM_PROMPT
        + "\n\nConversation:\n"
        + conversation
        + "\n\nJARVIS:"
    )

    try:
        retry_options = types.HttpRetryOptions(
            attempts=5,
            initial_delay=1.0,
            max_delay=16.0,
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        )

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=retry_options
            ),
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=700
            )
        )

        reply = response.text

        if not reply:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        logger.info("JARVIS response successful.")
        return {"reply": reply}

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Gemini request failed.")

        error_code = (
            getattr(exc, "code", None)
            or getattr(exc, "status_code", None)
        )

        if error_code == 503:
            raise HTTPException(
                status_code=503,
                detail="JARVIS is temporarily busy. Please try again shortly."
            )

        raise HTTPException(
            status_code=502,
            detail="JARVIS Gemini service encountered an error."
        )
