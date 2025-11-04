"""
Generador de respuestas. Soporta Ollama (local) o Azure OpenAI Foundry (chat/completions).
Se selecciona Azure si están definidas las variables de entorno
`AZURE_OPENAI_ENDPOINT` (URL completa al endpoint de chat/completions) y
`AZURE_OPENAI_KEY` (clave/API key). Si no están, se usa Ollama como antes.
"""
from typing import List, Dict, Optional
import sys
from pathlib import Path
import os
import requests

# Agregar utils al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.context_compressor import compress_context


class CNTGenerator:

    def __init__(self, model: str = "qwen2.5:7b", temperature: float = 0.1, max_context_chars: int = 8000, debug: bool = False, text_field: str = "texto"):

        self.model = model
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.debug = debug
        self.text_field = text_field  # Ahora usa 'texto' del retriever simplificado

        # Detectar configuración de Azure OpenAI (Foundry)
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_key = os.getenv("AZURE_OPENAI_KEY")
        # Si ambas están configuradas, usaremos Azure. En caso contrario, intentamos usar Ollama
        self.use_azure = bool(self.azure_endpoint and self.azure_key)

        if self.use_azure:
            if self.debug:
                print(f"Usando Azure OpenAI endpoint: {self.azure_endpoint}")
        else:
            # intentamos importar ollama solo si no hay Azure configurado
            try:
                import ollama
                self.ollama = ollama
                # comprobar que Ollama esté corriendo
                try:
                    self.ollama.list()
                except Exception as e:
                    raise RuntimeError("Ollama no está corriendo. Ejecuta: brew services start ollama o configura AZURE_OPENAI_*") from e
            except Exception:
                # Si falla la importación, informar al usuario
                raise RuntimeError("No hay endpoint de Azure configurado y no se pudo inicializar Ollama. Configura AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_KEY o instala/ejecuta Ollama.")
    
    def _resolve_text(self, article: Dict) -> str:
        """
        Resuelve el texto del artículo con fallback múltiple.
        Intenta varios campos en orden de prioridad para máxima compatibilidad.
        
        Args:
            article: Diccionario con datos del artículo
            
        Returns:
            Texto del artículo o string vacío si no se encuentra
        """
        # Fallback múltiple: probar varios campos comunes
        text = (article.get(self.text_field)
                or article.get("content")         # Compresor
                or article.get("texto")            # Alternativa común
                or article.get("texto_completo")   # Campo estándar
                or article.get("body")             # Genérico
                or article.get("snippet")          # Extracto
                or "")
        
        return text.strip() if text else ""

    # Método para formatear el contexto de los artículos
    def _format_context(self, articles: List[Dict]) -> str:

        context_parts = []
        
        for article in articles:
            # Resolver número de artículo con fallback
            art_num = (article.get('articulo_completo')
                      or article.get('articulo')
                      or article.get('id')
                      or 'N/A')
            
            # Resolver texto con fallback múltiple
            text = self._resolve_text(article)
            
            if text:  # Solo agregar si hay contenido
                part = f"ARTÍCULO {art_num}\n{text}"
                context_parts.append(part)
        
        return "\n\n---\n\n".join(context_parts)
    
    # Método principal para generar respuesta
    def generate(self, query: str, articles: List[Dict], system_prompt: Optional[str] = None) -> Dict:
        
        # Comprimir contexto si es necesario (artículos largos)
        compressed_articles = compress_context(articles, query, max_total_chars=self.max_context_chars)
        context = self._format_context(compressed_articles)
        
        if system_prompt is None:
            system_prompt = """Eres un asistente del Código Nacional de Tránsito de Colombia (Ley 769/2002).

PROCESO OBLIGATORIO ANTES DE RESPONDER:

1. LEE COMPLETO cada artículo proporcionado
2. IDENTIFICA toda la información relevante (incluyendo excepciones, pasos permitidos, condiciones especiales)
3. ANALIZA si hay detalles que modifiquen la regla general
4. CONSTRUYE la respuesta incluyendo TODA la información pertinente

REGLAS DE RESPUESTA:

SI la pregunta NO es sobre tránsito:
→ "Solo puedo responder sobre el Código Nacional de Tránsito. ¿Tienes alguna consulta sobre normas de tránsito?"

SI NO encuentras la información en los artículos:
→ "No encontré información sobre eso en el CNT."

SI encuentras la información:
→ Respuesta COMPLETA en 2-3 oraciones
→ INCLUYE excepciones y condiciones especiales si existen
→ Cita el artículo exacto
→ Agrega multa si aplica

FRASES PROHIBIDAS:
❌ "según el texto proporcionado"
❌ "el artículo menciona"
❌ "esto implica que"

EJEMPLOS DE ANÁLISIS COMPLETO:

P: "¿Puedo girar a la derecha en rojo?"
❌ MAL: "Las señales luminosas indican detenerse en rojo (Artículo 118)."
✅ BIEN: "Sí, el giro a la derecha en luz roja está permitido respetando la prelación del peatón, salvo que haya señalización especial prohibiéndolo (Artículo 118)."

P: "¿Puedo pasar en amarillo?"
❌ MAL: "El amarillo indica atención (Artículo 118)."
✅ BIEN: "No, no debes ingresar en amarillo a la intersección. Es infracción grave con multa de 30 SMMLV (Artículos 118 y 129-D)."

P: "¿Límite de velocidad en ciudad?"
❌ MAL: "Máximo 50 km/h en vías urbanas (Artículo 106)."
✅ BIEN: "Máximo 50 km/h en vías urbanas, 30 km/h en zonas escolares y residenciales (Artículo 106)."

CRÍTICO: Lee TODO el artículo antes de responder. No omitas excepciones ni condiciones especiales.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"ARTÍCULOS DEL CNT:\n{context}\n\n---\n\nPREGUNTA: {query}\n\nINSTRUCCIÓNES: Lee COMPLETO cada artículo. Identifica excepciones y condiciones especiales. Responde en 2-3 oraciones incluyendo TODA la información relevante.\n\nRESPUESTA:"}
        ]
        
        # Si está configurado Azure, hacemos la petición HTTP al endpoint proporcionado
        if self.use_azure:
            headers = {
                # Azure suele aceptar api-key; algunos endpoints preview pueden aceptar Authorization Bearer.
                # Añadimos ambos para aumentar compatibilidad.
                "Authorization": f"Bearer {self.azure_key}",
                "api-key": self.azure_key,
                "Content-Type": "application/json"
            }

            # Construir payload compatible con chat/completions
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                # Ajuste sensato de tokens; puede adaptarse según necesidades
                "max_tokens": 1024
            }

            try:
                r = requests.post(self.azure_endpoint, headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                response = r.json()
            except Exception as e:
                raise RuntimeError(f"Error llamando a Azure OpenAI endpoint: {e}") from e

            # Parsear la respuesta atendiendo a variantes de formato
            answer = ""
            try:
                # Forma común: choices[0].message.content
                if "choices" in response and len(response["choices"]) > 0:
                    first = response["choices"][0]
                    if isinstance(first.get("message"), dict) and "content" in first.get("message"):
                        answer = first["message"]["content"]
                    elif "message" in first and isinstance(first.get("message"), str):
                        answer = first.get("message")
                    elif "content" in first:
                        answer = first.get("content")
                    else:
                        # intentar otras claves
                        answer = str(first)
                else:
                    # fallback genérico
                    answer = response.get("text", "") or response.get("response", "") or ""
            except Exception:
                answer = ""

            usage = response.get("usage", {}) if isinstance(response, dict) else {}

            return {
                "answer": answer,
                "model": self.model,
                "articles_used": [a.get('articulo') for a in articles],
                "usage": usage
            }

        # Fallback: Ollama local (mismo comportamiento anterior)
        else:
            response = self.ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': 250,  # Suficiente para respuestas completas con excepciones
                    'num_ctx': 4096,
                    'num_thread': 8
                }
            )

            answer = response['message']['content']

            return {
                "answer": answer,
                "model": self.model,
                "articles_used": [a.get('articulo') for a in articles],
                "usage": {
                    "prompt_tokens": response.get('prompt_eval_count', 0),
                    "completion_tokens": response.get('eval_count', 0),
                    "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                }
            }
