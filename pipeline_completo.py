"""
Pipeline completo del sistema RAG para el CNT.
Genera embeddings e índice FAISS desde el PDF.
"""
import sys
import time
from pathlib import Path

# Agregar src al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "preprocessing"))
sys.path.insert(0, str(project_root / "src" / "chunking"))
sys.path.insert(0, str(project_root / "src" / "embeddings"))
sys.path.insert(0, str(project_root / "src"))

from pdf_extractor import PDFExtractor
from text_cleaner import TextCleaner
from chunker import ArticleChunker
from embedding_generator import EmbeddingGenerator


class CNTPipeline:
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.pdf_path = self.project_root / "data" / "raw" / "cnt.pdf"
        self.raw_text_path = self.project_root / "data" / "processed" / "cnt_raw.txt"
        self.clean_text_path = self.project_root / "data" / "processed" / "cnt_clean.txt"
        self.chunks_path = self.project_root / "data" / "processed" / "cnt_chunks.json"
        self.index_dir = self.project_root / "data" / "index"
        
    def step1_preprocessing(self):
        print("\n[1/3] Preprocesamiento del PDF")
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {self.pdf_path}")
        
        print(f"  Extrayendo texto de {self.pdf_path.name}")
        extractor = PDFExtractor(str(self.pdf_path))
        raw_text = extractor.extract_text(str(self.raw_text_path))
        
        print("  Limpiando texto")
        cleaner = TextCleaner()
        clean_text = cleaner.clean_all(raw_text)
        
        self.clean_text_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.clean_text_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        
        print(f"  Guardado: {len(clean_text):,} caracteres")
        return clean_text
    
    def step2_chunking(self):
        print("\n[2/3] Chunking de articulos")
        
        if not self.clean_text_path.exists():
            raise FileNotFoundError(f"Texto limpio no encontrado: {self.clean_text_path}")
        
        chunker = ArticleChunker(str(self.clean_text_path))
        chunks = chunker.chunk_text(str(self.chunks_path))
        
        print(f"  Total de chunks: {len(chunks)}")
        return chunks
    
    def step3_embeddings(self):
        print("\n[3/3] Generacion de embeddings e indice FAISS")
        
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks no encontrados: {self.chunks_path}")
        
        print("  Esto puede tomar 1-2 minutos...")
        
        generator = EmbeddingGenerator(
            model_name='intfloat/multilingual-e5-base',
            batch_size=64,
            use_query_prefix=True
        )
        
        result = generator.process(
            chunks_file=str(self.chunks_path),
            out_dir=str(self.index_dir)
        )
        
        print(f"  Total: {result['total_chunks']} chunks")
        print(f"  Dimension: {result['embedding_dimension']}")
        
        return result
    
    def step4_validation(self):
        print("\nValidando archivos generados...")
        
        archivos_necesarios = [
            self.index_dir / "embeddings.npy",
            self.index_dir / "meta.jsonl",
            self.index_dir / "faiss.index"
        ]
        
        todo_ok = True
        for archivo in archivos_necesarios:
            existe = archivo.exists()
            status = "OK" if existe else "FALTA"
            print(f"  [{status}] {archivo.name}")
            if not existe:
                todo_ok = False
        
        if todo_ok:
            print("\nSistema listo. Iniciar bot con: python whatsapp_bot.py")
        else:
            print("\nError: Algunos archivos no se generaron")
        
        return todo_ok
    
    def run(self, skip_preprocessing=False, skip_chunking=False):
        print("Pipeline CNT - Generacion de embeddings")
        print(f"Directorio: {self.project_root}")
        
        start_time = time.time()
        
        try:
            if skip_preprocessing:
                print("\nSaltando preprocesamiento")
            else:
                self.step1_preprocessing()
            
            if skip_chunking:
                print("\nSaltando chunking")
            else:
                self.step2_chunking()
            
            self.step3_embeddings()
            self.step4_validation()
            
            elapsed = time.time() - start_time
            print(f"\nTiempo total: {elapsed:.1f} segundos")
            
        except Exception as e:
            print(f"\nError: {e}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup inicial del sistema RAG')
    parser.add_argument('--skip-preprocessing', action='store_true')
    parser.add_argument('--skip-chunking', action='store_true')
    
    args = parser.parse_args()
    
    pipeline = CNTPipeline()
    pipeline.run(
        skip_preprocessing=args.skip_preprocessing,
        skip_chunking=args.skip_chunking
    )


if __name__ == "__main__":
    main()
