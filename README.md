# Product Matching Service

Python service that extracts key phrases from Amazon product URLs using AWS Comprehend and ranks local vendor products by similarity. Results are persisted to PostgreSQL. Includes a Flask API and a CLI entry.

## Features
- AWS Comprehend key-phrase extraction from Amazon URLs
- TF-IDF + cosine similarity via scikit-learn (fallback to keyword overlap)
- Results persisted to `comparable_items` in PostgreSQL (top 150)
- Flask endpoint `POST /find-matching-products` and CLI in `main.py`

## Setup
1. Python deps:
   - `pip install -r requirements.txt`
   - `python -m nltk.downloader punkt` (if needed for stemming)
2. AWS creds for Comprehend available in your environment (e.g., via AWS CLI)
3. PostgreSQL:
   - Apply `db/schema.sql`
   - Ensure `products` and `business` tables exist as referenced by queries

## Configuration
Configure DB via environment variables (defaults in parentheses):
- `DB_NAME` (Company Database)
- `DB_USER` (postgres)
- `DB_PASSWORD` (QHacks)
- `DB_HOST` (localhost)
- `DB_PORT` (5432)

## Run
- API: `python flask.py` then `POST /find-matching-products` with `{ "url": "<amazon_url>" }`
- CLI: edit `amazon_url` in `main.py` or pass via code, then `python main.py`

