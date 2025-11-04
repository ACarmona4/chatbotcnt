"""
Embeddings + índice FAISS para los chunks creados
"""
import json, numpy as np
from pathlib import Path
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import faiss

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

# Método para cargar chunks
def load_chunks(path: str) -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["articulos"]


class EmbeddingGenerator:

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", batch_size: int = 64, use_query_prefix: bool = True):

        # Elegimos device automáticamente (cuda si está disponible)
        self.model = SentenceTransformer(model_name, device="cuda" if HAS_CUDA else "cpu")
        self.batch_size = batch_size
        self.model_name = model_name
        self.use_query_prefix = use_query_prefix
        # flag para compatibilidad con prefijos tipo 'e5'
        self.is_e5_model = "e5" in model_name.lower()

    # Método para agregar prefijo 'passage:' a los textos
    def _to_passage(self, text: str) -> str:
        if not self.use_query_prefix:
            return text
        return f"passage: {text.strip()}" if not text.startswith("passage:") else text

    # Método para agregar prefijo 'query:' a las consultas
    def _to_query(self, text: str) -> str:
        if not self.use_query_prefix:
            return text
        return f"query: {text.strip()}" if not text.startswith("query:") else text

    # Método para generar embeddings de passages
    def encode_passages(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return np.asarray(embs, dtype=np.float32)

    # Método para generar embedding de una query
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        query_with_prefix = self._to_query(query)
        emb = self.model.encode(
            [query_with_prefix],
            batch_size=1,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return np.asarray(emb[0], dtype=np.float32)

    # Método para preparar texto para embedding (extraer full_text)
    def _prepare_text_for_embedding(self, obj: Dict) -> str:
        return obj.get('full_text', '').strip()

    # Método para extraer metadata del chunk
    def _extract_metadata(self, obj: Dict, idx: int) -> Dict:
        metadata_obj = obj.get("metadata", {})
        
        article_num = obj["article_number"]
        article_num_full = obj["article_number_full"]
        title = obj["title"]
        text = obj["full_text"]
        
        # Contexto jerárquico
        titulo_numero = metadata_obj.get("titulo_numero", "")
        titulo_nombre = metadata_obj.get("titulo_nombre", "")
        capitulo_numero = metadata_obj.get("capitulo_numero", "")
        capitulo_nombre = metadata_obj.get("capitulo_nombre", "")
        
        # Construir contexto
        context_parts = []
        if titulo_numero and titulo_nombre:
            context_parts.append(f"Título {titulo_numero}: {titulo_nombre}")
        if capitulo_numero and capitulo_nombre:
            context_parts.append(f"Capítulo {capitulo_numero}: {capitulo_nombre}")
        context_parts.append(f"Artículo {article_num_full}")
        if title:
            context_parts.append(title)
        
        context = " | ".join(context_parts)
        
        return {
            "doc_id": idx + 1,
            "articulo": article_num,
            "articulo_completo": article_num_full,
            "titulo": title,
            "titulo_numero": titulo_numero,
            "titulo_nombre": titulo_nombre,
            "capitulo_numero": capitulo_numero,
            "capitulo_nombre": capitulo_nombre,
            "longitud_caracteres": metadata_obj.get("longitud_caracteres", 0),
            "longitud_palabras": metadata_obj.get("longitud_palabras", 0),
            "tiene_paragrafos": metadata_obj.get("tiene_paragrafos", False),
            "tiene_modificaciones": metadata_obj.get("tiene_modificaciones", False),
            "texto_completo": text,
            "contexto": context
        }

    # Método principal para procesar chunks y crear índice FAISS
    def process(self, chunks_file: str, out_dir: str = "data/index") -> Dict:
        
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        # Cargar chunks
        chunks = load_chunks(chunks_file)
        
        metas, passages = [], []
        for idx, obj in enumerate(chunks):
            text = self._prepare_text_for_embedding(obj)
            if not text:
                continue
            
            meta = self._extract_metadata(obj, idx)
            metas.append(meta)
            passages.append(self._to_passage(text))

        # Generar embeddings
        embs = self.encode_passages(passages, normalize=True)

        # Guardar archivos
        np.save(out / "embeddings.npy", embs)
        
        with (out / "meta.jsonl").open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        # Crear índice FAISS
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        faiss.write_index(index, str(out / "faiss.index"))

        return {
            "total_chunks": len(passages),
            "embedding_dimension": dim,
            "model_name": self.model_name,
            "index_path": str(out / "faiss.index")
        }

# Buscador con índice FAISS
class FAISSSearcher:
    
    def __init__(self, index_dir: str, model_name: str = "intfloat/multilingual-e5-base"):
        self.index_dir = Path(index_dir)
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        
        # Cargar metadata
        self.metas = []
        with (self.index_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))
        
        self.embedder = EmbeddingGenerator(model_name=model_name, use_query_prefix=True)

    # Método para buscar chunks relevantes
    def search(self, query: str, top_k: int = 5) -> List[Dict]:

        query_emb = self.embedder.encode_query(query, normalize=True)
        query_emb = np.expand_dims(query_emb, axis=0)
        
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metas):
                result = self.metas[idx].copy()
                result["score"] = float(score)
                results.append(result)
        
        return results