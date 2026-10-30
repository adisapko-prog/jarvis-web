
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are JARVIS, an intelligent personal AI assistant.
Your personality is composed, sophisticated, observant,
calm, articulate, quietly witty and helpful.

Speak naturally and confidently. Be warm but professional.
Explain complicated things clearly and step by step.
Never pretend to have abilities you do not possess.

You have no human face or body. Your visual identity
is a futuristic blue-cyan holographic Arc Reactor.
"""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.get("/")
def home():
    return {"status": "JARVIS online"}

@app.post("/api/chat")
def chat(data: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "AI key is not configured.")

    if not data.messages or data.messages[-1].role != "user":
        raise HTTPException(400, "Please send a user message.")

    if any(m.role not in ["user", "assistant"] for m in data.messages):
        raise HTTPException(400, "Invalid message role.")

    client = OpenAI(api_key=api_key)

    try:
        result = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            instructions=SYSTEM_PROMPT,
            input=[
                {"role": m.role, "content": m.content[:4000]}
                for m in data.messages[-12:]
            ],
            max_output_tokens=500
        )
        return {"reply": result.output_text}

    except Exception:
        raise HTTPException(502, "JARVIS AI service is temporarily unavailable.")
