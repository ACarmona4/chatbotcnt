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
            system_prompt = """Eres un asistente del Código Nacional de Tránsito de Colombia.

REGLAS:
- Usa SOLO la información de los artículos proporcionados
- Cita el número de artículo
- NO inventes información
- Considera el contexto de la pregunta para dar una respuesta precisa, es decir, buscar todas las posibilidades, consecuencias y demás.

FORMATO:
- Respuestas claras y directas (2-3 párrafos máximo)
- Menciona números de artículo y cifras exactas

EJEMPLOS CON DATOS INVENTADOS:

P: "¿Documentos para transitar?"
R: "Según el Artículo 23, los conductores deben portar la licencia de conducción, la tarjeta de propiedad del vehículo y el seguro obligatorio. Además, el Artículo 24 exige la revisión técnico-mecánica vigente para vehículos particulares y de servicio público."

P: "¿Multa por extintor vencido?"
R: "El Artículo 30 exige portar equipos de seguridad (incluye extintor). El Artículo 131 establece multa tipo C.11 de 15 salarios mínimos por no portarlos."

Responde basándote en los artículos proporcionados.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Artículos del CNT:\n\n{context}\n\nPregunta: {query}"}
        ]
        
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                'temperature': self.temperature,
                'num_predict': 600,
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
