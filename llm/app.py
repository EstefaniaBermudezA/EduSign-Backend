import os
import time
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_TIMEOUT_S = float(os.getenv("HF_TIMEOUT_S", "60.0"))

if not HF_TOKEN:
    raise RuntimeError("Falta HF_TOKEN en variables de entorno.")

app = FastAPI(title="HF Chat Proxy (OpenAI-compatible)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    prompt: str = Field(..., description="Pregunta del niño")
    character: str | None = Field(
        None,
        description="Personaje histórico que responde (ej. 'Anubis'). Si se omite, responde un narrador neutro.",
    )
    model: str | None = Field(None, description="Override del modelo (opcional)")
    max_tokens: int = Field(150, ge=1, le=512, description="Límite de tokens de salida")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Temperatura de muestreo")

class AskResponse(BaseModel):
    model: str
    answer: str
    latency_ms: int

@app.get("/health")
def health():
    return {"ok": True, "provider_base": OPENAI_BASE_URL, "model": MODEL_ID}

def build_system_prompt(character: str | None) -> str:
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
    """
    Proxy a la API OpenAI-compatible de Hugging Face (/v1/chat/completions).
    Genera una respuesta breve para mostrar como texto en UE5 tras el botón "saber más".
    """
    api_url = f"{OPENAI_BASE_URL}/chat/completions"
    model = req.model or MODEL_ID
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt(req.character)},
            {"role": "user", "content": req.prompt}
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=HF_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
    except requests.Timeout as e:
        raise HTTPException(status_code=504, detail=f"Timeout > {HF_TIMEOUT_S}s: {e}")
    except requests.HTTPError as e:
        # Propaga error del proveedor para depurar
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail={"error": str(e), "provider": detail})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    dt_ms = int((time.perf_counter() - t0) * 1000)
    return AskResponse(model=model, answer=content, latency_ms=dt_ms)
