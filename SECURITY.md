# Security Policy - KrishiMitra AI

## Supported Versions

Version 1.0.x - Active support
Version below 1.0 - No support

## Responsible AI Security

### Data Privacy
- No farmer PII stored (name, phone, location)
- Images processed in memory only, not saved to disk
- Audit logs contain only disease and confidence data
- No user tracking or analytics
- ChromaDB contains only ICAR public documents

### API Security
- Rate limiting: 10 requests per minute per IP
- File type validation on image upload
- File size limit: 10MB maximum
- Input sanitization on all text fields
- CORS configured for production

### Known Limitations
- Model confidence can be low for unseen crop varieties
- Fallback advisory used when IBM API unavailable
- Demo model trained on synthetic data only

## Reporting a Vulnerability

DO NOT open a public GitHub issue for security vulnerabilities.

Email: krishimitra.ai@gmail.com
Subject: [SECURITY] Brief description

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix if known

## Response Timeline

Acknowledgment: Within 24 hours
Assessment: Within 72 hours
Critical fix: 7 days
High fix: 14 days
Medium fix: 30 days
Low fix: Next release

## Security Best Practices for Contributors

Never commit .env file - use .env.example instead
Check for secrets before pushing:
  grep -r "api_key" app/ --include=*.py
Keep dependencies updated:
  pip install --upgrade -r requirements.txt
