
import os
import logging

import openai
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JARVIS")

# Application
app = FastAPI(title="JARVIS AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# JARVIS personality
SYSTEM_PROMPT = """
You are JARVIS, an advanced personal AI assistant.

PERSONALITY:
- Intelligent, composed and sophisticated.
- Calm, articulate and quietly witty.
- Warm, helpful and professional.
- Observant and thoughtful.
- Speak naturally, not like a robot.

BEHAVIOUR:
- Understand the user's intent before responding.
- Explain complicated things clearly.
- Give step-by-step instructions when needed.
- Be concise when a short answer is enough.
- Never invent facts or pretend to have abilities
  you do not possess.
- Remember the conversation context provided to you.

IDENTITY:
You are JARVIS.
You have no human face or physical body.
Your visual identity is a futuristic blue-cyan
holographic Arc Reactor inspired by advanced technology.
"""

# Request models
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=30)


# Health check
@app.get("/")
def home():
    return {"status": "JARVIS online"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# Chat endpoint
@app.post("/api/chat")
def chat(data: ChatRequest):

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")

    if not api_key:
        logger.error("OPENAI_API_KEY is missing.")
        raise HTTPException(
            status_code=503,
            detail="JARVIS AI configuration is incomplete."
        )

    if data.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="The last message must be from the user."
        )

    client = OpenAI(
        api_key=api_key,
        timeout=45.0,
        max_retries=1
    )

    conversation = [
        {
            "role": message.role,
            "content": message.content
        }
        for message in data.messages[-12:]
    ]

    try:
        logger.info("JARVIS request started. Model: %s", model)

        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=conversation,
            max_output_tokens=700
        )

        reply = response.output_text

        if not reply:
            logger.error("OpenAI returned an empty response.")
            raise HTTPException(
                status_code=502,
                detail="JARVIS returned an empty response."
            )

        logger.info(
            "JARVIS response successful. Request ID: %s",
            response._request_id
        )

        return {"reply": reply}

    except HTTPException:
        raise

    except openai.AuthenticationError as e:
        logger.error("OpenAI authentication failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail="JARVIS authentication failed. Check the API key."
        )

    except openai.RateLimitError as e:
        logger.error("OpenAI rate limit or quota error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="JARVIS API quota or rate limit reached."
        )

    except openai.APIStatusError as e:
        logger.error(
            "OpenAI API error. Status: %s Request ID: %s Error: %s",
            e.status_code,
            e.request_id,
            e.message
        )
        raise HTTPException(
            status_code=502,
            detail="JARVIS encountered an OpenAI API error."
        )

    except openai.APIConnectionError as e:
        logger.error("OpenAI connection error: %s", e)
        raise HTTPException(
            status_code=502,
            detail="JARVIS could not connect to the AI service."
        )

    except openai.APIError as e:
        logger.error("OpenAI SDK error: %s", e)
        raise HTTPException(
            status_code=502,
            detail="JARVIS AI service encountered an error."
        )

    except Exception:
        logger.exception("Unexpected JARVIS backend error.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected JARVIS error occurred."
        )
