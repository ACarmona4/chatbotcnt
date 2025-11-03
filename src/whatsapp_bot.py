"""
Bot de WhatsApp para el Código Nacional de Tránsito (CNT)
Sistema RAG integrado con qwen2.5:7b + BM25 + Compresión
"""
import os
import json
import requests
import sys
import uvicorn
from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

# Agregar src al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from retriever.retriever import CNTRetriever
from generator.generator import CNTGenerator

load_dotenv()
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "cntc-secret")
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL", "https://graph.facebook.com/v22.0")
SEND_URL = f"{WHATSAPP_API_URL}/{PHONE_NUMBER_ID}/messages"

if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
    raise ValueError("Falta WHATSAPP_TOKEN o PHONE_NUMBER_ID en .env")

# Inicializar sistema RAG
try:
    base_dir = Path(__file__).parent.parent
    
    print("🔧 Inicializando CNTRetriever...")
    retriever = CNTRetriever(
        index_path=str(base_dir / "data" / "index" / "faiss.index"),
        meta_path=str(base_dir / "data" / "index" / "meta.jsonl"),
        model_name="intfloat/multilingual-e5-base",
        top_k=6,
        overfetch=32,
        min_score=0.35,
        use_bm25=True
    )
    print("✅ CNTRetriever inicializado")
    
    print("🔧 Inicializando CNTGenerator...")
    generator = CNTGenerator(
        model="qwen2.5:7b",
        temperature=0.1,
        max_context_chars=8000,
        debug=False
    )
    print("CNTGenerator inicializado")
    
except Exception as e:
    print(f"Error inicializando RAG: {e}")
    raise

app = FastAPI(title="CNT WhatsApp Bot", version="2.0.0")

# Cache para evitar procesar mensajes duplicados
processed_messages = set()
MAX_CACHE_SIZE = 1000


def send_whatsapp_text(to: str, body: str) -> Dict[str, Any]:
    """Envía un texto por WhatsApp. Divide en chunks si es muy largo."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    max_len = 4000
    chunks = [body[i:i+max_len] for i in range(0, len(body), max_len)] or ["(respuesta vacía)"]
    last = {}
    
    for i, part in enumerate(chunks, 1):
        if len(chunks) > 1:
            part = f"Parte {i}/{len(chunks)}\n\n{part}"
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": part}
        }
        
        try:
            r = requests.post(SEND_URL, headers=headers, json=payload, timeout=30)
            last = r.json()
        except Exception as e:
            last = {"error": str(e)}
    
    return last


def process_query(query: str) -> str:
    """Procesa una consulta con el sistema RAG integrado."""
    try:
        # 1. Recuperar artículos relevantes
        articles = retriever.search(query, top_k=6)
        
        if not articles:
            return "Lo siento, no encontré información relevante en el Código Nacional de Tránsito para tu consulta. 🔍"
        
        # 2. Generar respuesta con el LLM
        result = generator.generate(query, articles)
        answer = result.get("answer", "").strip()
        
        if not answer:
            return "Lo siento, no pude generar una respuesta. Intenta reformular tu pregunta."
        
        return answer
        
    except Exception as e:
        print(f"Error procesando query: {e}")
        return f"Ocurrió un error procesando tu consulta. Por favor, intenta de nuevo."


@app.get("/webhook")
async def verify_webhook(request: Request):
    mode      = request.query_params.get("hub.mode")
    token     = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Invalid verify token")


@app.post("/webhook")
async def inbound_message(request: Request):
    try:
        data = await request.json()
        changes = data["entry"][0]["changes"][0]["value"]
        messages = changes.get("messages", [])
        if not messages:
            return {"status": "no_message"}

        msg = messages[0]
        from_number = msg.get("from")
        mtype = msg.get("type")

        # Solo texto
        if mtype != "text":
            send_whatsapp_text(from_number, "Por ahora solo acepto mensajes de texto.")
            return {"status": "ignored_non_text"}

        user_text = (msg.get("text", {}) or {}).get("body", "").strip()
        if not user_text:
            send_whatsapp_text(from_number, "Mensaje vacío. Intenta de nuevo, por favor.")
            return {"status": "empty_text"}

        # Deduplicación de mensajes
        msg_id = msg.get("id")
        if msg_id in processed_messages:
            return {"status": "duplicate"}
        
        processed_messages.add(msg_id)
        if len(processed_messages) > MAX_CACHE_SIZE:
            processed_messages.clear()
        
        # Comando de ayuda
        if user_text.lower() in {"hola", "ayuda", "menu", "hi", "hello"}:
            help_text = (
                "👋 ¡Hola! Soy el asistente del Código Nacional de Tránsito (Ley 769/2002).\n\n"
                "Puedes preguntarme sobre:\n"
                "• Multas y sanciones\n"
                "• Documentos obligatorios\n"
                "• Normas de tránsito\n"
                "• Artículos específicos\n\n"
            )
            send_whatsapp_text(from_number, help_text)
            return {"status": "ok_help"}

        # Procesar consulta con RAG
        answer = process_query(user_text)
        send_whatsapp_text(from_number, answer)
        return {"status": "ok", "query": user_text, "answer_len": len(answer)}

    except Exception as e:
        # Nunca dejes al usuario sin respuesta…
        try:
            from_number = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
            send_whatsapp_text(from_number, "Hubo un error procesando tu mensaje. Intenta de nuevo en unos minutos.")
        except Exception:
            pass
        return {"status": "error", "detail": str(e)}


@app.get("/")
async def root():
    return {
        "service": "CNT WhatsApp Bot",
        "version": "2.0.0",
        "model": generator.model,
        "retriever": "CNTRetriever + BM25",
        "status": "ok"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": generator.model}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")