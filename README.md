# Chatbot CNT - Asistente Virtual del Código Nacional de Tránsito

## Introducción

Este proyecto implementa un asistente virtual inteligente para el Código Nacional de Tránsito de Colombia (Ley 769/2002), utilizando técnicas de Retrieval-Augmented Generation (RAG). El sistema permite a los usuarios realizar consultas en lenguaje natural a través de WhatsApp y obtener respuestas precisas basadas en la normativa mencionada.

El sistema combina búsqueda semántica avanzada, re-ranking de resultados y generación de respuestas mediante modelos de lenguaje de última generación, proporcionando información confiable sobre multas, sanciones, requisitos de documentación y normas de tránsito.

## Tecnologías Utilizadas

### Backend y Framework
- **Python 3.13**: Lenguaje de programación principal
- **FastAPI**: Framework web para la API REST
- **Uvicorn**: Servidor ASGI de alto rendimiento

### Procesamiento de Lenguaje Natural
- **Ollama**: Runtime local para modelos de lenguaje
- **Qwen2.5:7b**: Modelo de lenguaje para generación de respuestas
- **Sentence Transformers**: Generación de embeddings semánticos
- **FAISS**: Búsqueda vectorial eficiente
- **BM25**: Algoritmo de ranking para búsqueda léxica

### Integración y Comunicación
- **WhatsApp Business API**: Plataforma de mensajería
- **ngrok**: Túnel HTTP para desarrollo local
- **requests**: Cliente HTTP para integraciones

### Procesamiento de Documentos
- **PyPDF2**: Extracción de texto desde PDF
- **pdfplumber**: Procesamiento avanzado de documentos PDF

## Instalación

### Requisitos Previos
- Python 3.13 o superior
- Ollama instalado
- Cuenta de WhatsApp Business
- ngrok para exposición de webhook

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/ACarmona4/chatbotcnt.git
cd chatbotcnt
```

2. Crear y activar entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Instalar y configurar Ollama:
```bash

ollama pull qwen2.5:7b
```

5. Configurar variables de entorno:
```bash
cp .env.example .env
```

Editar `.env` con las credenciales de WhatsApp Business:
```
WHATSAPP_TOKEN=tu_token_de_whatsapp
PHONE_NUMBER_ID=tu_phone_number_id
VERIFY_TOKEN=cntc-secret
```

6. Generar índice de búsqueda (primera vez):
```bash
python scripts/pipeline_completo.py
```

## Ejecución

### Modo Desarrollo (con logs visibles)

Terminal 1 - Servidor del bot:
```bash
source .venv/bin/activate
python src/whatsapp_bot.py
```

Terminal 2 - Túnel ngrok:
```bash
ngrok http 8000
```

### Obtener URL pública del webhook

```bash
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

### Detener servicios

```bash
pkill -f whatsapp_bot.py
pkill ngrok
```

## Estructura del Proyecto

```
proyectofinal/
├── data/
│   ├── raw/                    # PDF original del CNT
│   ├── processed/              # Textos procesados y chunks
│   └── index/                  # Embeddings e índice FAISS
├── scripts/
│   └── pipeline_completo.py    # Pipeline de procesamiento RAG
├── src/
│   ├── chunking/               # Segmentación por artículos
│   │   └── chunker.py
│   ├── embeddings/             # Generación de embeddings
│   │   └── embedding_generator.py
│   ├── generator/              # Generación de respuestas con LLM
│   │   └── generator.py
│   ├── preprocessing/          # Extracción y limpieza de PDF
│   │   ├── pdf_extractor.py
│   │   └── text_cleaner.py
│   ├── retriever/              # Sistema de recuperación híbrido
│   │   └── retriever.py
│   ├── utils/                  # Utilidades y compresión
│   │   └── context_compressor.py
│   └── whatsapp_bot.py         # Aplicación principal
├── .env                        # Variables de entorno
├── .gitignore                  # Archivos excluidos de git
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación
```

### Contribuciones

1. Fork del repositorio
2. Crear rama feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit de cambios: `git commit -m 'Descripción del cambio'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## Autores

**Alejandro Carmona**
- GitHub: [@ACarmona4](https://github.com/ACarmona4)

**Sebastián Castaño Arroyave**
- GitHub: [@scastanoa1](https://github.com/scastanoa1)

**Miguel Alejandro Gómez Duque**
- GitHub: [@Magdc](https://github.com/Magdc)

