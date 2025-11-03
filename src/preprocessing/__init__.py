"""
Módulo de preprocesamiento para el Código Nacional de Tránsito Colombiano.

Este módulo contiene herramientas para:
- Extraer texto de archivos PDF
- Limpiar y normalizar texto extraído
- Pipeline completo de preprocesamiento

Estructura de datos:
- data/raw/: Contiene el archivo PDF original (cnt.pdf)
- data/processed/: Contiene los archivos procesados (cnt_raw.txt, cnt_clean.txt)
"""
from .pdf_extractor import PDFExtractor
from .text_cleaner import TextCleaner
from .pipeline import PreprocessingPipeline

__all__ = ['PDFExtractor', 'TextCleaner', 'PreprocessingPipeline']
__version__ = '1.0.0'
