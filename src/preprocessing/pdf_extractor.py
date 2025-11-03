"""
Módulo para extraer texto del PDF del Código Nacional de Tránsito.
"""
from pathlib import Path
import pdfplumber


class PDFExtractor:
    def __init__(self, pdf_path: str):
        
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")

    # Método para extraer texto del PDF
    def extract_text(self, output_path: str = None) -> str:

        full_text = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
        
        extracted_text = "\n".join(full_text)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
        
        return extracted_text