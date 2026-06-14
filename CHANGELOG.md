# Changelog - KrishiMitra AI

## [1.0.0] - 2026-06-14 - Initial Release

### Added
- EfficientNet-B0 CNN classifier for crop disease detection
- 38 PlantVillage disease classes supported
- RAG pipeline with ChromaDB vector database
- ICAR knowledge base with 8 documents in English and Hindi
- IBM Granite LLM integration for advisory generation
- Rule-based fallback advisory works without IBM API
- FastAPI REST backend with 3 endpoints
- Streamlit bilingual frontend English and Hindi
- Rate limiting 10 requests per minute
- Confidence-based expert escalation threshold 0.75
- Audit logging for Responsible AI compliance
- 85 unit and integration tests
- Docker deployment support
- HuggingFace Spaces deployment
- Demo dataset generator for 5 crop disease classes
- Model training script with early stopping
- Model evaluation script with per-class accuracy

### Tech Stack
- Python 3.11
- FastAPI 0.111.0
- Streamlit 1.35.0
- PyTorch 2.3.0
- ChromaDB 1.5.9
- SentenceTransformers 2.7.0

### Known Limitations
- Model trained on synthetic demo data only
- Real PlantVillage dataset needed for 95%+ accuracy
- IBM Granite requires paid API key
- Only tomato and potato in demo training set

## [Upcoming] - v1.1.0

### Planned
- Rice disease detection
- Wheat disease detection
- Tamil language support
- Telugu language support
- WhatsApp Bot integration
- Weather-based disease risk alerts
- Offline mode for rural areas
- Voice query input
