---
title: Cocokin AI API
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Cocokin AI API

FastAPI service for CV parsing, job recommendation, and Gemini-powered recommendation explanations.

## Endpoints

- `GET /health`
- `POST /recommend-from-cv?use_gemini=true`
- `POST /analyze-target-role?target_role=Data Scientist`

Set `GEMINI_API_KEY` and `GEMINI_MODEL` in Hugging Face Spaces secrets.
