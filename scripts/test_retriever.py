"""Prueba rápida del retriever: carga índice y busca queries de ejemplo.
"""
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.retriever.retriever import CNTRetriever

retriever = CNTRetriever(
    index_path=str(project_root / "data" / "index" / "faiss.index"),
    meta_path=str(project_root / "data" / "index" / "meta.jsonl"),
    model_name="sentence-transformers/all-mpnet-base-v2",
    top_k=5,
    overfetch=32,
    min_score=0.2,
    use_bm25=True,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

queries = [
    "¿Puedo girar a la derecha en rojo?",
    "¿Cuál es la multa por pasar en amarillo?",
    "Límite de velocidad en zona escolar"
]

for q in queries:
    print("\n=== Query:\n", q)
    res = retriever.search(q)
    for r in res:
        print(f"score={r['score']:.4f} articulo={r.get('articulo')} titulo={r.get('titulo')}")
        print(r['texto'][:200].replace('\n', ' '))
        print('---')
