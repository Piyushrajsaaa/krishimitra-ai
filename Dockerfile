FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit==1.35.0 \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    Pillow \
    sentence-transformers==2.7.0 \
    chromadb \
    python-dotenv \
    pydantic==2.7.1 \
    pydantic-settings \
    requests \
    sqlalchemy==2.0.30 \
    aiofiles \
    httpx \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    python-multipart==0.0.9 \
    slowapi==0.1.9

# Copy app code
COPY . .

# Run ingestion script to build knowledge base
RUN python scripts/ingest_knowledge.py

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
