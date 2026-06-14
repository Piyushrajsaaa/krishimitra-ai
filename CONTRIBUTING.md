# Contributing to KrishiMitra AI

Thank you for contributing to KrishiMitra AI!
This project helps Indian farmers detect crop diseases using AI.
SDG 2 - Zero Hunger | 1M1B AI for Sustainability

## Table of Contents
1. Code of Conduct
2. Development Setup
3. Project Structure
4. Coding Standards
5. Testing Guidelines
6. Submitting Changes
7. Issue Guidelines
8. Priority Areas
9. Common Errors

## Code of Conduct
- Be respectful and inclusive
- Focus on farmer welfare
- Accept constructive criticism
- Show empathy to all contributors
- Never discriminate based on language or region

## Development Setup

### Prerequisites
- Python 3.11+
- Git 2.30+
- RAM 8GB+
- Disk 5GB+

### Step 1 - Fork and Clone
git clone https://github.com/YOUR_USERNAME/krishimitra-ai.git
cd krishimitra-ai

### Step 2 - Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

### Step 3 - Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

### Step 4 - Configure Environment
cp .env.example .env

### Step 5 - Setup Knowledge Base
python scripts/ingest_knowledge.py

### Step 6 - Setup Dataset
python scripts/download_dataset.py

### Step 7 - Verify Setup
pytest tests/ -v
# Expected: 85+ tests passing

### Step 8 - Run Application
# Terminal 1
uvicorn app.main:app --reload --port 8000
# Terminal 2
streamlit run app/frontend.py --server.port 8501

## Project Structure

krishimitra-ai/
  app/
    api/routes/
      advisory.py       - POST /advisory endpoint
      health.py         - GET /health endpoint
    core/
      config.py         - Settings from .env
      logger.py         - App and audit logging
    models/
      disease_classifier.py  - EfficientNet-B0
    schemas/
      advisory.py       - Pydantic models
    services/
      llm_service.py    - IBM Granite + fallback
      rag_service.py    - ChromaDB retrieval
    utils/
      image_preprocessor.py  - Image to tensor
    frontend.py         - Streamlit UI
    main.py             - FastAPI entry point
  data/
    knowledge_base/
      icar_knowledge.json    - ICAR documents
      chroma_db/             - Vector database
    processed/               - Training dataset
  models/checkpoints/        - Trained weights
  scripts/
    download_dataset.py
    ingest_knowledge.py
    train_model.py
    evaluate_model.py
  tests/
    unit/
      test_health.py
      test_schemas.py
      test_model.py
      test_rag.py
      test_llm.py
    integration/
      test_api.py

## Coding Standards

- Use type hints on all functions
- Write docstrings for all public functions
- Keep functions under 50 lines
- Use descriptive variable names
- Follow PEP 8 style guide
- Use f-strings not .format()
- Log events using logger.info()
- Always handle exceptions with try/except

Example of good code:
def preprocess_image(
    image_bytes: bytes,
    training: bool = False
) -> Optional[torch.Tensor]:
    Converts raw image bytes to model-ready tensor.
    Args:
        image_bytes: Raw bytes from file upload
        training: If True, applies augmentation
    Returns:
        Tensor [1, 3, 224, 224] or None if invalid

## Testing Guidelines

Run all tests:
  pytest tests/ -v

Run specific file:
  pytest tests/unit/test_model.py -v

Run with coverage:
  pytest tests/ --cov=app --cov-report=html

Requirements for new tests:
- Minimum 80% code coverage on new code
- Test happy path, edge cases, and error handling
- Use mocks for external services (IBM, ChromaDB)
- Tests must run without internet connection
- Tests must complete in under 30 seconds

Current Test Coverage:
  test_health.py      - 5 tests
  test_schemas.py     - 8 tests
  test_model.py       - 16 tests
  test_rag.py         - 14 tests
  test_llm.py         - 20 tests
  test_api.py         - 22 tests
  TOTAL               - 85 tests

## Submitting Changes

Branch naming:
  fix/issue-123-description
  feat/add-wheat-disease
  docs/improve-setup-guide
  test/add-rag-edge-cases

Commit message format:
  feat: add wheat disease detection
  fix: resolve chromadb timeout
  docs: add hindi translation guide
  test: add image preprocessor edge cases

Steps:
1. git checkout -b feat/your-feature
2. Make changes
3. pytest tests/ -v (must pass)
4. git add .
5. git commit -m "feat: your description"
6. git push origin feat/your-feature
7. Create Pull Request on GitHub

## Issue Guidelines

Bug Report must include:
- OS and Python version
- Steps to reproduce
- Expected vs actual behavior
- Error logs from logs/krishimitra.log
- Screenshots if UI issue

Feature Request must include:
- Problem description
- Proposed solution
- Farmer impact estimate
- SDG alignment
- Willingness to implement

## Priority Areas

HIGH PRIORITY:
1. More Indian crops - Rice, Wheat, Cotton, Sugarcane
2. Regional languages - Tamil, Telugu, Bengali, Marathi
3. WhatsApp integration
4. Offline mode for rural areas
5. Voice input for low literacy farmers

MEDIUM PRIORITY:
6. More ICAR documents
7. Weather integration
8. Pest detection
9. Yield prediction
10. Expert escalation to KVK officers

GOOD FIRST ISSUES:
11. Fix typos in advisory text
12. Add unit tests for edge cases
13. Improve error messages
14. Add loading animations
15. Improve Hindi translations

## Common Errors and Solutions

Error 1: ChromaDB not found
  Fix: python scripts/ingest_knowledge.py

Error 2: Model checkpoint missing
  Fix: python scripts/train_model.py

Error 3: IBM API not configured
  This is normal for development.
  Get free IBM key at: https://cloud.ibm.com/registration

Error 4: Port already in use
  Fix: lsof -ti:8000 | xargs kill -9

Error 5: CUDA errors
  Code runs on CPU by default. No GPU needed.

Error 6: Import errors after pulling
  Fix: pip install -r requirements.txt --upgrade

Error 7: Tests failing
  Never change DISEASE_CLASSES list order.
  Add new classes only at the end.

## Getting Help

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Questions and ideas
- Email: Security issues only

Response Times:
- Bug reports: 48 hours
- Feature requests: 1 week
- Pull requests: 72 hours
- Security issues: 24 hours

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in academic citations
- Eligible for 1M1B volunteer recognition

Thank you for helping Indian farmers!
Every contribution makes a difference.
