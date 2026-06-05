"""Servicio HTTP que expone un LLM como narrador/personaje histórico para EduSign.

Define una API FastAPI con dos endpoints: ``/health`` (sonda de estado) y ``/ask``,
que reenvía la pregunta de un niño a un modelo de Hugging Face (vía el router con
formato OpenAI) y devuelve una respuesta corta y simple en español. Requiere la
variable de entorno ``HF_TOKEN``. Se ejecuta con: ``uvicorn app:app --reload``.
"""

import os
import time
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise RuntimeError("Falta HF_TOKEN en variables de entorno.")

OPENAI_BASE_URL = "https://router.huggingface.co/v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_TIMEOUT_S = 60.0

app = FastAPI(title="EduSign LLM", version="1.0.0")


class AskRequest(BaseModel):
    """Cuerpo de la petición al endpoint ``/ask``."""

    prompt: str = Field(..., description="Pregunta del niño")
    character: str | None = Field(
        None,
        description="Personaje histórico que responde (ej. 'Anubis'). Si se omite, responde un narrador neutro.",
    )
    max_tokens: int = Field(150, ge=1, le=512)
    temperature: float = Field(0.3, ge=0.0, le=2.0)


class AskResponse(BaseModel):
    """Respuesta del endpoint ``/ask``: texto generado y latencia del servidor."""

    answer: str
    latency_ms: int


@app.get("/health")
def health():
    """Comprueba que el servicio está vivo e indica el modelo configurado."""
    return {"ok": True, "model": MODEL_ID}


def build_system_prompt(character: str | None) -> str:
    """Construye el system prompt que fija persona, idioma y estilo apto para niños.

    Args:
        character: Personaje histórico a encarnar; si es ``None`` se usa un
            narrador educativo neutro.
    """
    persona = (
        f"Eres {character}, un personaje histórico, y hablas en primera persona."
        if character
        else "Eres un narrador educativo amable."
    )
    return (
        f"{persona} "
        "Le respondes a un niño sordo que está aprendiendo. "
        "Reglas obligatorias: responde SIEMPRE en español, usa 2 o 3 oraciones cortas, "
        "vocabulario muy simple (evita palabras raras, metáforas e ironía), "
        "información correcta y apropiada para niños. "
        "No hagas preguntas de vuelta, no saludes, ve directo a la respuesta."
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Responde la pregunta de un niño en boca de un personaje histórico.

    Reenvía el prompt al LLM de Hugging Face y devuelve una respuesta breve en
    español junto con la latencia medida en el servidor.

    Raises:
        HTTPException: 504 si el proveedor excede el timeout, el código de estado
            del proveedor ante un error HTTP, o 500 para cualquier otro fallo.
    """
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": build_system_prompt(req.character)},
            {"role": "user", "content": req.prompt},
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=HF_TIMEOUT_S,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
    except requests.Timeout as e:
        raise HTTPException(status_code=504, detail=f"Timeout > {HF_TIMEOUT_S}s: {e}")
    except requests.HTTPError as e:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail={"error": str(e), "provider": detail})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        answer=content,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
