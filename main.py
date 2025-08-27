from urllib.parse import urlparse
import boto3
import psycopg2
import os
import re
from nltk.stem import PorterStemmer
from datetime import datetime
from similarity import rank_by_tfidf_cosine, sklearn_available

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

# Preprocess text by making everything lowercase and removing special characters
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]  # Apply stemming using nltk library
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


# Get product names, descriptions, prices, and seller names from the database
def fetch_product_data():
    try:
        # Connect to the PostgreSQL database
        connection = get_db_conn()
        cursor = connection.cursor()

        # Fetch product details including seller
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
    
    #If server is not properly connected or the database fails, return nothing
    except Exception as e:
        print(f"Error fetching product data: {e}")
        return []

# Split key phrases into individual words
def split_key_phrases_into_words(key_phrases):
    words = []
    for phrase in key_phrases:
        words.extend(preprocess_text(phrase).split())  # Preprocess and split
    return list(set(words))  # Return unique words

# Calculate similarity score
def calculate_similarity_score(words, name_words, desc_words):
    # Count matches between the input words and product words
    matched_words = set(words) & set(name_words + desc_words)
    return len(matched_words)

# Find matching products based on similarity score
def _find_matching_products_keyword(words, product_data):
    matching_products = []
    for p in product_data:
        name_words = preprocess_text(p["product_name"]).split()
        desc_words = preprocess_text(p["description"]).split()
        score = calculate_similarity_score(words, name_words, desc_words)
        if score > 0:
            matching_products.append(
                (p["product_name"], p["description"], p["price"], p["seller"], float(score), p["product_id"])  # include id
            )
    matching_products.sort(key=lambda x: x[-2], reverse=True)
    return matching_products


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
                    results.append(
                        (p["product_name"], p["description"], p["price"], p["seller"], float(score), p["product_id"])  # include id
                    )
            results.sort(key=lambda x: x[-2], reverse=True)
            return results
        except Exception:
            pass
    return _find_matching_products_keyword(words, product_data)


def persist_matches(amazon_url: str, matches: list[tuple], limit: int = 150) -> int:
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
                    m[-1],  # product_id
                    m[-2],  # similarity_score
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

# Main function
def main(amazon_url):
    # Extract the product name from the URL
    product_name = extract_product_name(amazon_url)
    if not product_name:
        print("Could not extract product name from the URL.")
        return

    print(f"Extracted Product Name: {product_name}")

    # Analyze the product name using AWS Comprehend
    key_phrases = analyze_product_name(product_name)
    print(f"Extracted Key Phrases: {key_phrases}")

    # Split key phrases into individual words
    words = split_key_phrases_into_words(key_phrases)
    print(f"Split Key Phrases into Words: {words}")

    # Fetch product data from the database
    product_data = fetch_product_data()

    # Find matching products
    matching_products = find_matching_products(words, product_data)

    # Display the matching products or a message if none are found
    if not matching_products:
        print("\nNo matching products found.")
    else:
        print("\nMatching Products:")
        for name, desc, price, seller, score, _pid in matching_products:
            print(f"Product: {name}")
            print(f"Price: ${price:.2f}")
            print(f"Sold by: {seller}")
            print(f"Description: {desc}")
            print(f"Similarity Score: {score}\n")

        # Persist top 150 matches
        inserted = persist_matches(amazon_url, matching_products, limit=150)
        print(f"Persisted {inserted} comparable items to Postgres.")


if __name__ == "__main__":
    amazon_url = "https://www.amazon.ca/Scented-Candles-Fireside-Aromatherapy-Masculine/dp/B08WCCQZBN/?_encoding=UTF8&pd_rd_w=qFVHt&content-id=amzn1.sym.058704f3-b5a4-43fe-a9c9-166bf808d15b%3Aamzn1.symc.a68f4ca3-28dc-4388-a2cf-24672c480d8f&pf_rd_p=058704f3-b5a4-43fe-a9c9-166bf808d15b&pf_rd_r=0BA46XG8195HBYR768NE&pd_rd_wg=mVy8P&pd_rd_r=dfad7154-d342-472a-afc7-734d78534f5c&ref_=pd_hp_d_atf_ci_mcx_mr_ca_hp_atf_d&th=1"
    main(amazon_url)
