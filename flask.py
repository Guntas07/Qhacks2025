from flask import Flask, request, jsonify
from urllib.parse import urlparse
import boto3
import psycopg2
import os
import re
from nltk.stem import PorterStemmer
from datetime import datetime
from similarity import rank_by_tfidf_cosine, sklearn_available

# Initialize Flask app
app = Flask(__name__)

# Initialize stemmer
stemmer = PorterStemmer()

# AWS Comprehend client
comprehend = boto3.client('comprehend', region_name='us-east-1')

# Extract the product name from an Amazon URL
def extract_product_name(amazon_url):
    parsed_url = urlparse(amazon_url)
    path_parts = parsed_url.path.split('/')
    for part in path_parts:
        if part and 'dp' not in part and 'gp' not in part:
            return part.replace('-', ' ')

# Analyze product name with AWS Comprehend
def analyze_product_name(product_name):
    response = comprehend.detect_key_phrases(
        Text=product_name,
        LanguageCode='en'
    )
    key_phrases = [phrase['Text'] for phrase in response['KeyPhrases']]
    return key_phrases

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# DB connection helper
def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "Company Database"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "QHacks"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )


# Fetch product data from the database
def fetch_product_data():
    try:
        connection = get_db_conn()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT 
                p.product_id,
                p.product_name, 
                p.description, 
                p.price, 
                b.business_name
            FROM products p
            JOIN business b ON p.business_id = b.business_id;
        """)
        rows = cursor.fetchall()
        # Normalize into list of dicts for clarity
        product_data = [
            {
                "product_id": r[0],
                "product_name": r[1],
                "description": r[2],
                "price": r[3],
                "seller": r[4],
            }
            for r in rows
        ]
        cursor.close()
        connection.close()
        return product_data
    except Exception as e:
        print(f"Error fetching product data: {e}")
        return []

# Split key phrases into words
def split_key_phrases_into_words(key_phrases):
    words = []
    for phrase in key_phrases:
        words.extend(preprocess_text(phrase).split())
    return list(set(words))

def _find_matching_products_keyword(words, product_data):
    matching_products = []
    for p in product_data:
        name_words = preprocess_text(p["product_name"]).split()
        desc_words = preprocess_text(p["description"]).split()
        score = len(set(words) & set(name_words + desc_words))
        if score > 0:
            matching_products.append({
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "description": p["description"],
                "price": p["price"],
                "seller": p["seller"],
                "similarity_score": float(score),
            })
    matching_products.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matching_products


# Find matching products using scikit-learn TF-IDF cosine similarity when available
def find_matching_products(words, product_data):
    if sklearn_available():
        query_text = " ".join(words)
        product_texts = [
            preprocess_text(f"{p['product_name']} {p['description']}") for p in product_data
        ]
        try:
            sims = rank_by_tfidf_cosine(query_text, product_texts)
            results = []
            for p, score in zip(product_data, sims):
                if score > 0:
                    results.append({
                        "product_id": p["product_id"],
                        "product_name": p["product_name"],
                        "description": p["description"],
                        "price": p["price"],
                        "seller": p["seller"],
                        "similarity_score": float(score),
                    })
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results
        except Exception:
            # Fall back if vectorization fails for any reason
            pass
    # Keyword overlap fallback
    return _find_matching_products_keyword(words, product_data)


def persist_matches(amazon_url: str, matches: list[dict], limit: int = 150) -> int:
    """Persist top-N matches into comparable_items table. Returns number inserted."""
    if not matches:
        return 0
    to_insert = matches[:limit]
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.executemany(
            (
                "INSERT INTO comparable_items (amazon_url, product_id, similarity_score, created_at) "
                "VALUES (%s, %s, %s, %s)"
            ),
            [
                (
                    amazon_url,
                    m["product_id"],
                    m["similarity_score"],
                    datetime.utcnow(),
                )
                for m in to_insert
            ],
        )
        conn.commit()
        cur.close()
        conn.close()
        return len(to_insert)
    except Exception as e:
        print(f"Error persisting matches: {e}")
        return 0

# REST API endpoint
@app.route('/find-matching-products', methods=['POST'])
def find_matching_products_api():
    try:
        # Get JSON payload
        data = request.json
        amazon_url = data.get("url")

        if not amazon_url:
            return jsonify({"error": "No URL provided"}), 400

        # Extract product name
        product_name = extract_product_name(amazon_url)
        if not product_name:
            return jsonify({"error": "Invalid Amazon URL"}), 400

        # Analyze product name
        key_phrases = analyze_product_name(product_name)
        words = split_key_phrases_into_words(key_phrases)

        # Fetch product data
        product_data = fetch_product_data()
        if not product_data:
            return jsonify({"error": "No product data available"}), 500

        # Find matching products
        matching_products = find_matching_products(words, product_data)

        # Persist top results to Postgres (up to 150)
        try:
            inserted = persist_matches(amazon_url, matching_products, limit=150)
        except Exception as _:
            inserted = 0

        return jsonify({
            "matching_products": matching_products,
            "persisted_count": inserted,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
