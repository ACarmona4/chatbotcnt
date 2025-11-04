"""
CNTRetriever (versión simple)
- Búsqueda semántica con FAISS (e5)
- Filtro por evidencia mínima
- Deduplicación por artículo
- (Opcional) Reorden BM25
- (Opcional) Recuperación directa por "art. N"

Requisitos:
pip install sentence-transformers faiss-cpu
# opcional: pip install rank-bm25
"""
from __future__ import annotations
import json, re
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
try:
    # CrossEncoder puede usarse como re-ranker para ordenar mejor los candidatos
    from sentence_transformers import CrossEncoder
except Exception:
    try:
        # fallback a otra ruta de import
        from sentence_transformers.cross_encoder import CrossEncoder
    except Exception:
        CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False


class CNTRetriever:
    def __init__(
        self,
        index_path: str = "data/index/faiss.index",
        meta_path: str  = "data/index/meta.jsonl",
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        top_k: int      = 6,
        overfetch: int  = 32,
        min_score: float = 0.35,
        use_bm25: bool   = False,
        text_fields_priority: Optional[List[str]] = None,
        rerank_model: Optional[str] = None
    ):
        # modelo (GPU si existe; transparente)
        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in model_name.lower()

        # índice FAISS y metadatos (misma longitud)
        self.index = faiss.read_index(index_path)
        self.meta: List[Dict] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))
        assert self.index.ntotal == len(self.meta), "FAISS y meta.jsonl desalineados"

        # parámetros
        self.top_k = top_k
        self.overfetch = max(top_k, overfetch)
        self.min_score = float(min_score)

        # campos de texto (auto-fallback)
        self.text_fields_priority = text_fields_priority or [
            "texto", "texto_completo", "content", "body", "snippet"
        ]

        # mapa de artículo → idx (para acceso directo si viene “art. N”)
        self.art_to_idx = {}
        for i, m in enumerate(self.meta):
            a = m.get("articulo")
            if isinstance(a, int):
                self.art_to_idx[a] = i

        # BM25 opcional
        self.use_bm25 = bool(use_bm25 and HAS_BM25)
        self._bm25 = None
        if self.use_bm25:
            tokens_corpus = [self._tokenize(self._get_text(m)) for m in self.meta]
            self._bm25 = BM25Okapi(tokens_corpus)

        # Re-ranker opcional (cross-encoder). Si se pasa rerank_model lo cargamos.
        self.reranker = None
        if rerank_model and CrossEncoder is not None:
            try:
                self.reranker = CrossEncoder(rerank_model)
            except Exception:
                self.reranker = None

    # ------------- utils -------------
    def _prefix_query(self, q: str) -> str:
        return f"query: {q}" if self.is_e5 else q

    def _embed_query(self, q: str) -> np.ndarray:
        return self.model.encode(
            [self._prefix_query(q)],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

    def _get_text(self, m: Dict) -> str:
        for k in self.text_fields_priority:
            t = m.get(k)
            if t:
                return t
        return ""

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    @staticmethod
    def _extract_article_numbers(query: str) -> List[int]:
        q = query.lower()
        nums = set()
        for pat in (r"art[íi]culo\s+(\d+)", r"art\.?\s+(\d+)", r"articulo\s+(\d+)"):
            for m in re.findall(pat, q):
                try:
                    nums.add(int(m))
                except:
                    pass
        return sorted(nums)

    # ------------- search -------------
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        k = top_k or self.top_k
        candidates: List[Dict] = []
        seen_articles = set()

        # A) acceso directo si mencionan “art. N” (sin heurísticas extra)
        for num in self._extract_article_numbers(query):
            idx = self.art_to_idx.get(num)
            if idx is not None:
                m = self.meta[idx]
                candidates.append(self._pack(m, score=1.0, source="direct"))
                seen_articles.add(num)

        # B) vectorial (overfetch) + filtro por evidencia
        q_emb = self._embed_query(query)
        D, I = self.index.search(q_emb, self.overfetch)
        for idx, s in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            art = m.get("articulo")
            if art in seen_articles:  # dedup por artículo
                continue
            if s < self.min_score:
                continue
            candidates.append(self._pack(m, score=float(s), source="semantic"))
            seen_articles.add(art)

        # Si existe reranker, re-evaluar scores de los candidatos y reordenar
        if self.reranker and len(candidates) > 0:
            # Limitarnos a los candidatos actuales
            texts = [self._get_text(c) for c in candidates]
            pairs = [(query, t) for t in texts]
            try:
                rerank_scores = self.reranker.predict(pairs)
                # Reemplazar score por la puntuación del reranker (normalizamos a 0-1 con sigmoid si es necesario)
                for c, s in zip(candidates, rerank_scores):
                    try:
                        c["score"] = float(s)
                    except Exception:
                        pass
            except Exception:
                # Si reranker falla, continuar con scores originales
                pass

        # C) BM25 opcional: reordenar candidatos por léxico (mejora definiciones)
        if self.use_bm25 and self._bm25 and len(candidates) > 1:
            docs = [self._get_text(c) for c in candidates]
            q_tokens = self._tokenize(query)
            bm_scores = self._bm25.get_scores(q_tokens)
            
            # Obtener scores solo para los índices de los candidatos
            candidate_bm_scores = []
            for c in candidates:
                doc_id = c.get("doc_id")
                # Buscar el índice en meta que corresponde a este doc_id
                idx = next((i for i, m in enumerate(self.meta) if m.get("doc_id") == doc_id), None)
                if idx is not None and idx < len(bm_scores):
                    candidate_bm_scores.append(bm_scores[idx])
                else:
                    candidate_bm_scores.append(0.0)
            
            # Combina: 70% semántico, 30% BM25 (simple, estable)
            for c, bm in zip(candidates, candidate_bm_scores):
                c["score"] = 0.7 * c["score"] + 0.3 * (bm / 10.0)

        # D) orden final y recorte
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    # ------------- pack -------------
    def _pack(self, m: Dict, score: float, source: str) -> Dict:
        return {
            "score": score,
            "source": source,
            "doc_id": m.get("doc_id"),
            "articulo": m.get("articulo"),
            "titulo": m.get("titulo"),
            "titulo_nombre": m.get("titulo_nombre"),
            "capitulo": m.get("capitulo"),
            "capitulo_nombre": m.get("capitulo_nombre"),
            "epigrafe": m.get("epigrafe"),
            "texto": self._get_text(m)
        }