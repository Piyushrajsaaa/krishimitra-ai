# 🌱 KrishiMitra AI
> **Your AI farming expert — anytime, anywhere**
> आपका AI कृषि विशेषज्ञ — कभी भी, कहीं भी

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-85%20passing-brightgreen)]()
[![SDG](https://img.shields.io/badge/SDG-2%20Zero%20Hunger-orange)]()

---

## Problem Statement

India loses **₹50,000+ crore annually** due to undetected crop diseases.
**600M+ farmers** lack timely access to agricultural experts.
Rural KVKs are understaffed and distant.

> *How might we use AI to detect crop diseases from leaf images and deliver
> real-time, plain-language advisory so that Indian farmers can act immediately
> and reduce crop loss sustainably?*

---

## Solution

KrishiMitra AI is a **production-grade, multimodal AI system** that:

1.  Accepts a crop leaf photo from any smartphone
2.  Detects disease using **EfficientNet-B0** (fine-tuned on PlantVillage)
3.  Retrieves relevant **ICAR knowledge** via RAG (ChromaDB)
4.  Generates **farmer-friendly advisory** in English & Hindi
5.  Collects feedback for continuous improvement

---

##  System Architecture
Farmer (Mobile/Web)

│

▼

┌─────────────────┐

│  Streamlit UI   │  ← Language toggle, image upload, feedback

└────────┬────────┘

│ HTTP

▼

┌─────────────────┐

│   FastAPI API   │  ← /advisory, /feedback, /health

└────────┬────────┘

│

┌────┴────┐

▼         ▼

┌────────┐ ┌──────────────┐

│ CNN    │ │ RAG Pipeline │

│ Model  │ │ ChromaDB     │

│ Eff-B0 │ │ ICAR Docs    │

└────┬───┘ └──────┬───────┘

│             │

└──────┬──────┘

▼

┌───────────────┐

│  LLM Service  │

│  IBM Granite  │

│  (+ Fallback) │

└───────────────┘

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.35 |
| API | FastAPI + Uvicorn |
| CNN Model | EfficientNet-B0 (PyTorch) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB 1.5.9 |
| LLM | IBM Granite (+ rule-based fallback) |
| Knowledge Base | ICAR agricultural documents |
| Testing | pytest (85 tests) |

---

##  Project Structure
krishimitra-ai/

├── app/

│   ├── api/routes/       # FastAPI routes (health, advisory)

│   ├── core/             # Config, logger

│   ├── models/           # EfficientNet-B0 classifier

│   ├── schemas/          # Pydantic request/response models

│   ├── services/         # RAG service, LLM service

│   ├── utils/            # Image preprocessor

│   └── frontend.py       # Streamlit UI

├── data/

│   ├── knowledge_base/   # ICAR documents + ChromaDB

│   └── processed/        # Training dataset

├── models/checkpoints/   # Trained model weights

├── scripts/              # Training, evaluation, ingestion

├── tests/                # 85 unit + integration tests

├── logs/                 # App + audit logs

└── requirements.txt
---

##  Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/krishimitra-ai.git
cd krishimitra-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your IBM API key
```

### 3. Setup Knowledge Base
```bash
python scripts/download_dataset.py
python scripts/ingest_knowledge.py
```

### 4. Train Model
```bash
python scripts/train_model.py
```

### 5. Start API Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Launch Frontend
```bash
streamlit run app/frontend.py --server.port 8501
```

Visit: **http://localhost:8501**

---

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

**Current: 85 tests passing **

---

##  AI Features

| Feature | Details |
|---|---|
| Disease Detection | EfficientNet-B0, 82% accuracy (demo), 95%+ (full dataset) |
| Knowledge Retrieval | RAG with ICAR + KVK documents |
| Advisory Generation | IBM Granite LLM + rule-based fallback |
| Languages | English + Hindi |
| Confidence Threshold | 0.75 (escalates to KVK expert below) |

---

##  Responsible AI

-  Confidence-based escalation to human experts
-  No PII stored — audit logs are anonymized
-  Bilingual support for language fairness
-  Disclaimer on every AI-generated advisory
-  Full audit trail for every prediction
-  Graceful fallback when IBM unavailable

---

##  SDG Alignment

| SDG | How |
|---|---|
| **SDG 2** Zero Hunger | Reduces crop loss through early disease detection |
| **SDG 15** Life on Land | Promotes precise treatment, reduces pesticide overuse |
| **SDG 9** Innovation | AI-powered advisory accessible to all farmers |

---

##  Author
**Piyush Raj**
1M1B AI for Sustainability Internship (IBM SkillsBuild + AICTE)
---
##  License
Free to use for educational and research purposes.
