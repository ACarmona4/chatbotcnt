"""
Chunker que extrae cada artículo completo con toda su estructura jerárquica.
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Optional


class ArticleChunker:
    
    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {input_file}")

        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.full_text = f.read()
    
    # Método para detectar la estructura jerárquica del documento
    def detect_structure(self) -> Dict:
        structure = {
            'titulos': [],
            'capitulos': [],
            'secciones': []
        }
        
        # Detectar títulos
        titulo_pattern = r'TÍTULO\s+([IVXLCDM]+)\s*[:\-]?\s*\n\s*(.+?)(?=\n)'
        for match in re.finditer(titulo_pattern, self.full_text, re.MULTILINE):
            structure['titulos'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'posicion': match.start()
            })
        
        # Detectar capítulos
        capitulo_pattern = r'CAPÍTULO\s+([IVXLCDM]+)\s*[:\-]?\s*\n\s*(.+?)(?=\n)'
        for match in re.finditer(capitulo_pattern, self.full_text, re.MULTILINE):
            structure['capitulos'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'posicion': match.start()
            })
        
        # Detectar secciones
        seccion_pattern = r'SECCIÓN\s+([IVXLCDM]+)\s*[:\-]?\s*\n\s*(.+?)(?=\n)'
        for match in re.finditer(seccion_pattern, self.full_text, re.MULTILINE):
            structure['secciones'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'posicion': match.start()
            })
        
        return structure
    
    # Método para extraer artículos completos
    def extract_articles(self) -> List[Dict]:

        article_pattern = r'(?:^|\n)\s*(?:ARTÍCULO|Artículo)\s*(\d+(?:-\d+)?)[°oOªº]?\s*\.'
        matches = list(re.finditer(article_pattern, self.full_text, re.MULTILINE | re.IGNORECASE))
        
        if not matches:
            return []
        
        chunks = []
        
        for i, match in enumerate(matches):
            article_num = match.group(1)
            start = match.start()
            
            # Determinar el final del artículo
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(self.full_text)
            
            # Extraer todo el contenido del artículo
            article_text = self.full_text[start:end].strip()
            
            # Validar que no esté vacío
            if len(article_text) < 20:
                continue
            
            # Procesar el contenido
            processed = self._process_article_content(article_text, article_num)
            
            if processed:
                chunks.append(processed)
        
        return chunks

    # Método para procesar el contenido de un artículo
    def _process_article_content(self, article_text: str, article_num: str) -> Optional[Dict]:
        
        lines = [l.strip() for l in article_text.split('\n') if l.strip()]
        
        if not lines:
            return None
        
        # Extraer título
        first_line = lines[0]
        
        title_match = re.match(
            r'(?:ARTÍCULO|Artículo)\s*(\d+(?:-\d+)?)[°oOªº]?\s*\.\s*(.+)?', 
            first_line, 
            re.IGNORECASE
        )
        
        if title_match:
            title = title_match.group(2).strip() if title_match.group(2) else ''
        else:
            title = ''

        content_lines = lines[1:] if len(lines) > 1 else []
        content = self._format_legal_content(content_lines)
        full_text = f"Artículo {article_num}. {title}\n\n{content}".strip()
        
        # Convertir número de artículo a int
        try:
            article_num_int = int(article_num.split('-')[0])
        except ValueError:
            article_num_int = 0
        
        return {
            'article_number': article_num_int,
            'article_number_full': article_num,
            'title': title,
            'content': content,
            'full_text': full_text,
            'raw_content': article_text,
            'metadata': {
                'tipo': 'artículo',
                'numero': article_num_int,
                'numero_completo': article_num,
                'longitud_caracteres': len(full_text),
                'longitud_palabras': len(full_text.split()),
                'tiene_paragrafos': bool(re.search(r'Parágrafo', content, re.IGNORECASE)),
                'tiene_modificaciones': bool(re.search(r'Modificado por', content, re.IGNORECASE))
            }
        }
    
    def _format_legal_content(self, lines: List[str]) -> str:
        """
        Formatea contenido legal del CNT preservando:
        - Parágrafos y sus numeraciones
        - Incisos y literales (a), b), 1., 2., etc.)
        - Modificaciones ("Modificado por...")
        - Notas jurisprudenciales
        - Texto declarado EXEQUIBLE/INEXEQUIBLE
        """
        if not lines:
            return ""
        
        formatted_lines = []
        
        for line in lines:
            is_paragrafo = re.match(r'^Parágrafo', line, re.IGNORECASE)
            is_numbered = re.match(r'^\d+[\.\)]\s+', line)
            is_lettered = re.match(r'^[a-zA-Z][\.\)]\s+', line)
            is_modification = re.search(r'Modificado por|Adicionado por|Derogado por', line, re.IGNORECASE)
            is_jurisprudence = re.search(r'declarado.*EXEQUIBLE|declarado.*INEXEQUIBLE|jurisprudencia', line, re.IGNORECASE)
            is_note = line.startswith('Ver ') or line.startswith('NOTA:')
            
            # Agregar saltos de línea antes de elementos importantes
            if any([is_paragrafo, is_modification, is_jurisprudence, is_note]):
                if formatted_lines:
                    formatted_lines.append('\n')
            elif any([is_numbered, is_lettered]):
                # Solo agregar salto si no es continuación
                if formatted_lines and not formatted_lines[-1].endswith('\n'):
                    formatted_lines.append('\n')
            
            formatted_lines.append(line)
        
        content = ' '.join(formatted_lines)
        content = re.sub(r'\n{3,}', '\n\n', content) 
        
        return content.strip()
    
    # Método para enriquecer chunks con contexto jerárquico
    def enrich_with_structure(self, chunks: List[Dict], structure: Dict) -> List[Dict]:
        
        for chunk in chunks:
            # Buscar la posición del artículo en el texto original
            art_num = chunk.get('article_number_full', chunk['article_number'])
            article_pattern = f"(?:Artículo|ARTÍCULO)\\s*{re.escape(str(art_num))}[°oOªº]?\\."
            match = re.search(article_pattern, self.full_text, re.IGNORECASE)
            
            if not match:
                continue
            
            pos = match.start()
            
            # Encontrar el título más reciente
            titulo_actual = None
            for titulo in sorted(structure['titulos'], key=lambda x: x['posicion']):
                if titulo['posicion'] < pos:
                    titulo_actual = titulo
            
            if titulo_actual:
                chunk['metadata']['titulo_numero'] = titulo_actual['numero']
                chunk['metadata']['titulo_nombre'] = titulo_actual['nombre']
            
            # Encontrar el capítulo más reciente
            capitulo_actual = None
            for capitulo in sorted(structure['capitulos'], key=lambda x: x['posicion']):
                if capitulo['posicion'] < pos:
                    capitulo_actual = capitulo
            
            if capitulo_actual:
                chunk['metadata']['capitulo_numero'] = capitulo_actual['numero']
                chunk['metadata']['capitulo_nombre'] = capitulo_actual['nombre']
            
            # Encontrar la sección más reciente
            seccion_actual = None
            for seccion in sorted(structure['secciones'], key=lambda x: x['posicion']):
                if seccion['posicion'] < pos:
                    seccion_actual = seccion
            
            if seccion_actual:
                chunk['metadata']['seccion_numero'] = seccion_actual['numero']
                chunk['metadata']['seccion_nombre'] = seccion_actual['nombre']
        
        return chunks
    
    # Método principal para chunking y guardado
    def chunk_text(self, output_file: str, include_stats: bool = True, validate: bool = True) -> List[Dict]:
        
        structure = self.detect_structure()
        chunks = self.extract_articles()
        
        if validate:
            self._validate_chunks(chunks)
        
        chunks = self.enrich_with_structure(chunks, structure)
        
        # Ordenar por número de artículo
        chunks.sort(key=lambda x: (x['article_number'], x.get('article_number_full', '')))
        
        # Guardar resultado
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'metadata': {
                'total_articulos': len(chunks),
                'estructura': {
                    'titulos': len(structure['titulos']),
                    'capitulos': len(structure['capitulos']),
                    'secciones': len(structure['secciones'])
                },
                'fecha_procesamiento': str(Path(self.input_file).stat().st_mtime)
            },
            'articulos': chunks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        if include_stats:
            self._print_statistics(chunks)
        
        return chunks