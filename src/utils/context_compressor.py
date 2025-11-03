import re
from typing import List, Dict

SENT_SPLIT = re.compile(r'(?<=[\.\?\!])\s+')
WORD_PATTERN = re.compile(r'\b\w{3,}\b')

# Método para resolver el texto del artículo con múltiples fallbacks
def _resolve_article_text(article: Dict) -> str:

    # Prioridad: texto (retriever nuevo) > texto_completo > content > otros
    text = (article.get("texto")             # Campo del retriever simplificado
            or article.get("texto_completo")  # Compatibilidad con versión anterior
            or article.get("content")         # Compresor
            or article.get("body")            # Genérico
            or article.get("snippet")         # Extracto
            or "")
    
    return text.strip() if text else ""

# Método para puntuar la relevancia de una oración
def _score_sentence(sent: str, query_terms: List[str]) -> float:
    s = sent.lower()
    
    # Conteo de términos relevantes
    hits = sum(s.count(t) for t in query_terms if t)
    
    # Bonus por términos clave legales
    legal_bonus = 0
    if any(kw in s for kw in ['multa', 'sanción', 'infracción', 'prohib', 'oblig']):
        legal_bonus = 0.5
    if any(kw in s for kw in ['debe', 'deberá', 'podrá', 'será']):
        legal_bonus += 0.3
    
    # Penalización por oraciones muy cortas o muy largas
    L = len(sent)
    length_penalty = 1.0
    if L < 50:
        length_penalty = 0.5
    elif L > 300:
        length_penalty = 0.7
    
    return (hits + legal_bonus) * length_penalty

# Método para comprimir el artículo
def compress_article(article_text: str, query: str, max_chars: int = 1500) -> str:

    # Si ya es corto, retornar sin modificar
    if len(article_text) <= max_chars:
        return article_text
    
    # Extraer título/encabezado (primera línea con "ARTÍCULO")
    lines = article_text.split('\n')
    header = ""
    body_start = 0
    
    for i, line in enumerate(lines):
        if 'ARTÍCULO' in line.upper() or 'ART.' in line.upper():
            header = line.strip()
            body_start = i + 1
            break
    
    # Resto del texto
    body_text = '\n'.join(lines[body_start:])
    
    # Dividir en oraciones
    sentences = [s.strip() for s in SENT_SPLIT.split(body_text) if s.strip()]
    
    # Extraer términos de consulta
    q_terms = [w.lower() for w in WORD_PATTERN.findall(query)]
    
    if not sentences:
        return article_text[:max_chars]
    
    # Scoring y ordenamiento
    scored = [(s, _score_sentence(s, q_terms)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Construcción del resultado
    result_parts = []
    if header:
        result_parts.append(header)
        current_len = len(header) + 1
    else:
        current_len = 0
    
    # Agregar oraciones más relevantes
    selected_sentences = []
    for sent, score in scored:
        if current_len + len(sent) + 2 <= max_chars:
            selected_sentences.append(sent)
            current_len += len(sent) + 2
        if current_len >= max_chars * 0.9:
            break
    
    # Si tenemos oraciones, agregarlas al resultado
    if selected_sentences:
        result_parts.append(' '.join(selected_sentences))
    
    result = '\n'.join(result_parts)
    
    # Si aún es muy largo, truncar
    if len(result) > max_chars:
        result = result[:max_chars] + "..."
    
    return result

# Método para comprimir múltiples artículos
def compress_context(articles: List[Dict], query: str, max_total_chars: int = 8000) -> List[Dict]:

    # Calcular total usando fallback múltiple
    total_chars = sum(len(_resolve_article_text(art)) for art in articles)
    
    # Si el total ya es aceptable, no comprimir
    if total_chars <= max_total_chars:
        return articles
    
    # Calcular límite por artículo
    avg_limit = max_total_chars // len(articles) if articles else max_total_chars
    
    compressed = []
    for art in articles:
        # Obtener contenido del artículo con fallback múltiple
        content = _resolve_article_text(art)
        
        # Artículos cortos no necesitan compresión
        if len(content) <= avg_limit:
            compressed.append(art)
        else:
            # Comprimir artículo largo
            compressed_content = compress_article(content, query, max_chars=avg_limit)
            
            # Mantener estructura original pero con contenido comprimido
            compressed_art = art.copy()
            compressed_art['content'] = compressed_content  # Campo nuevo para contenido comprimido
            compressed_art['compressed'] = True  # Marcador para debugging
            compressed.append(compressed_art)
    
    return compressed