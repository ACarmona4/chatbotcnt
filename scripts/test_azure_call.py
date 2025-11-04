"""Prueba manual: carga .env y ejecuta CNTGenerator.generate
Imprime la respuesta y algunos metadatos (sin exponer la clave).
"""
import os
from pathlib import Path
import json

# Cargar .env manualmente (sin depender de python-dotenv)
env_path = Path(__file__).resolve().parents[1] / '.env'
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"')
            os.environ.setdefault(k, v)

# Importar el generador
import sys
from pathlib import Path

# Añadir la raíz del proyecto al path para poder importar el paquete `src`
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.generator.generator import CNTGenerator

# Instanciar el generador -> detectará AZURE_OPENAI_* y usará Azure
print("Inicializando CNTGenerator (esto usará Azure si AZURE vars están definidas)...")
try:
    g = CNTGenerator(model="gpt-4o-mini", temperature=0.1, max_context_chars=2000, debug=True)
except Exception as e:
    print("Error inicializando CNTGenerator:", e)
    raise

# Consulta de prueba y artículo mínimo
query = "¿Puedo girar a la derecha en rojo?"
articles = [
    {"articulo": "118", "texto": "El giro a la derecha en luz roja está permitido respetando la prelación del peatón, salvo que haya señalización especial prohibiéndolo (Artículo 118)."}
]

print("Enviando petición de prueba a la API...")
try:
    res = g.generate(query, articles)
    answer = res.get('answer', '')
    usage = res.get('usage', {})
    model = res.get('model')

    print('\n--- Resultado de la prueba ---')
    print('Modelo usado:', model)
    print('Respuesta (primeros 800 chars):')
    print(answer[:800])
    print('\n--- Uso / metadatos ---')
    try:
        print(json.dumps(usage, indent=2, ensure_ascii=False))
    except Exception:
        print(usage)
    print('--- Fin de prueba ---')

except Exception as e:
    print('Error durante la petición:', e)
    raise
