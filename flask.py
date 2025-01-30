from flask import Flask, request, jsonify
from urllib.parse import urlparse
import boto3
import psycopg2
import re
from nltk.stem import PorterStemmer

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

# Fetch product data from the database
def fetch_product_data():
    try:
        connection = psycopg2.connect(
            dbname="Company Database",
            user="postgres",
            password="QHacks",
            host="localhost",
            port="5432"
        )
        cursor = connection.cursor()
        cursor.execute("""
            SELECT 
                p.product_name, 
                p.description, 
                p.price, 
                b.business_name
            FROM products p
            JOIN business b ON p.business_id = b.business_id;
        """)
        product_data = cursor.fetchall()
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

# Find matching products based on similarity
def find_matching_products(words, product_data):
    matching_products = []
    for name, desc, price, seller in product_data:
        name_words = preprocess_text(name).split()
        desc_words = preprocess_text(desc).split()
        score = len(set(words) & set(name_words + desc_words))
        if score > 0:
            matching_products.append({
                "product_name": name,
                "description": desc,
                "price": price,
                "seller": seller,
                "similarity_score": score
            })
    matching_products.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matching_products

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

        return jsonify({"matching_products": matching_products}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
