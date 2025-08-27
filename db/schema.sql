-- Products table (assumed existing)
-- CREATE TABLE products (
--   product_id SERIAL PRIMARY KEY,
--   business_id INTEGER NOT NULL,
--   product_name TEXT NOT NULL,
--   description TEXT NOT NULL,
--   price NUMERIC(12,2) NOT NULL
-- );

-- Business table (assumed existing)
-- CREATE TABLE business (
--   business_id SERIAL PRIMARY KEY,
--   business_name TEXT NOT NULL
-- );

-- Comparable items produced by similarity matching
CREATE TABLE IF NOT EXISTS comparable_items (
  id SERIAL PRIMARY KEY,
  amazon_url TEXT NOT NULL,
  product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
  similarity_score NUMERIC NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comparable_items_amazon_url ON comparable_items(amazon_url);
CREATE INDEX IF NOT EXISTS idx_comparable_items_product_id ON comparable_items(product_id);
CREATE INDEX IF NOT EXISTS idx_comparable_items_created_at ON comparable_items(created_at);

