"""
Generador de respuestas con Ollama.
"""
from typing import List, Dict, Optional
import ollama
import sys
from pathlib import Path

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
        
        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError("Ollama no está corriendo. Ejecuta: brew services start ollama") from e
    
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
        
        response = ollama.chat(
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
