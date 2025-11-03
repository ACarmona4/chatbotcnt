"""
Módulo para limpiar y normalizar el texto extraído del PDF.
"""
import re
from pathlib import Path


class TextCleaner:
    def __init__(self, text: str = None):
        self.text = text

    # Método para eliminar contenido antes del primer artículo
    def remove_content_before_first_article(self, text: str = None) -> str:
        
        text = text if text is not None else self.text
        match = re.search(r'Artículo\s+1[°o]?\.', text, re.IGNORECASE)
        
        if match:
            text = text[match.start():]
        
        return text

    # Método para eliminar espacios en blanco excesivos
    def remove_extra_whitespace(self, text: str = None) -> str:

        text = text if text is not None else self.text
        text = re.sub(r' +', ' ', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text

    # Método para eliminar caracteres especiales
    def remove_special_characters(self, text: str = None, keep_basic: bool = True) -> str:
    
        text = text if text is not None else self.text
        
        if keep_basic:
            text = re.sub(r'[^\w\s.,;:!?¿¡()\-\"\'°ªº\n]', '', text, flags=re.UNICODE)
        else:
            text = re.sub(r'[^\w\s\n]', '', text, flags=re.UNICODE)
        
        return text
    
    # Método para normalizar saltos de línea
    def normalize_line_breaks(self, text: str = None) -> str:

        text = text if text is not None else self.text
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text
    
    # Método para corregir palabras separadas por guiones
    def fix_hyphenation(self, text: str = None) -> str:
        
        text = text if text is not None else self.text
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text

    # Método para eliminar marcadores de página
    def remove_page_markers(self, text: str = None) -> str:
        
        text = text if text is not None else self.text
        text = re.sub(r'LEY NÚMERO 769 del 6 de agosto de 2002 Hoja No\.\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'"?Por la cual se expide el Código Nacional de Tránsito Terrestre y se dictan\s*otras disposiciones"?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    # Método para unir el texto entre páginas
    def merge_pages(self, text: str = None) -> str:
        """
        Une líneas que fueron cortadas entre páginas.
        
        NOTA: remove_page_markers() ya fue llamado antes en clean_all(),
        no es necesario llamarlo nuevamente aquí.
        """
        text = text if text is not None else self.text
        
        lines = text.split('\n')
        merged_lines = []
        
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                merged_lines.append('')
                i += 1
                continue
            
            while i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                
                if not next_line:
                    break
                if re.search(r'[.!?;:]$', current_line):
                    break
                elif re.search(r'^(Artículo|ARTÍCULO|CAPÍTULO|TÍTULO|Parágrafo|PARÁGRAFO|LEY|\d+\.)', next_line, re.IGNORECASE):
                    break
                elif current_line.isupper() and len(current_line) < 100:
                    break
                if current_line.endswith('-'):
                    current_line = current_line[:-1] + next_line
                else:
                    current_line = current_line + ' ' + next_line
                
                i += 1
            
            merged_lines.append(current_line)
            i += 1
        
        return '\n'.join(merged_lines)
    
    # Método para normalizar artículos y estructura legal
    def normalize_articles(self, text: str = None) -> str:

        text = text if text is not None else self.text
        
        text = re.sub(r'(?i)art[ií]culo\s+(\d+[°o]?\.)', r'\n\nArtículo \1', text)
        text = re.sub(r'(?i)art\.?\s+(\d+[°o]?\.)', r'\n\nArtículo \1', text)

        # Normalizar CAPÍTULOS
        text = re.sub(r'(?i)cap[ií]tulo\s+([IVXivx]+|\d+)', r'\n\nCAPÍTULO \1', text)
        
        # Normalizar TÍTULOS
        text = re.sub(r'(?i)t[ií]tulo\s+([IVXivx]+|\d+)', r'\n\nTÍTULO \1', text)
        
        # Normalizar PARÁGRAFOS
        text = re.sub(r'(?i)par[áa]grafo\.?\s*(\d*[°o]?)', r'\n\nParágrafo \1', text)
        
        return text
    
    # Método para eliminar headers/footers
    def remove_headers_footers(self, text: str = None) -> str:

        text = text if text is not None else self.text
        text = re.sub(r'(?i)rep[úu]blica\s+de\s+colombia', '', text)
        text = re.sub(r'(?i)diario\s+oficial', '', text)
        
        return text
    
    # Método para eliminar referencias legislativas no deseadas
    def remove_legislative_references(self, text: str = None) -> str:

        text = text if text is not None else self.text
        text = re.sub(r'(?i)(modificado|adicionado|derogado)\s+por\s+la\s+Ley\s+\d+\s+de\s+\d{4}', '', text)
        
        return text
    
    # Método principal para limpiar todo el texto
    def clean_all(self, text: str = None) -> str:
        """
        Ejecuta todos los pasos de limpieza en orden optimizado.
        
        ORDEN:
        1. Contenido inicial y estructura
        2. Normalización de formato
        3. Limpieza de artefactos del PDF
        4. Unión de páginas
        5. Normalización de artículos
        6. Limpieza final de espacios/caracteres
        
        """
        text = text if text is not None else self.text
        
        # Paso 1: Encontrar inicio del contenido legal
        text = self.remove_content_before_first_article(text)
        
        # Paso 2: Normalizar formato básico
        text = self.normalize_line_breaks(text)
        
        # Paso 3: Limpiar artefactos del PDF
        text = self.remove_page_markers(text)  # Solo una vez aquí
        text = self.remove_headers_footers(text)
        
        # Paso 4: Corregir guiones y unir páginas
        text = self.fix_hyphenation(text)
        text = self.merge_pages(text)  # Ya no llama a remove_page_markers
        
        # Paso 5: Normalizar estructura de artículos
        text = self.normalize_articles(text)
        
        # Paso 6: Limpieza final
        text = self.remove_special_characters(text, keep_basic=True)
        text = self.remove_extra_whitespace(text)
        
        return text
