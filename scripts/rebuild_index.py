"""Regenerar embeddings y reindexar FAISS usando el mejor modelo de embeddings (por defecto mpnet).
Usage:
    python scripts/rebuild_index.py

Esto leerá `data/processed/cnt_chunks.json` y escribirá en `data/index/`:
- embeddings.npy
- meta.jsonl
- faiss.index
"""
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.embeddings.embedding_generator import EmbeddingGenerator

chunks_file = str(project_root / "data" / "processed" / "cnt_chunks.json")
print("Cargando chunks desde:", chunks_file)

eg = EmbeddingGenerator(model_name="sentence-transformers/all-mpnet-base-v2", batch_size=64, use_query_prefix=True)
res = eg.process(chunks_file, out_dir=str(project_root / "data" / "index"))
print("Proceso completado:")
print(res)
