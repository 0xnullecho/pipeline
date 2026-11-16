"""Read the single source catalogue and validate its paired image vectors."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = ROOT / 'data/synthetic-catalogue/product_catalogue_30000.csv'
EMBEDDINGS_PATH = ROOT / 'data/embeddings/product_embeddings.npy'
MODEL_NAME = 'google/siglip2-base-patch16-224'
CATEGORY_MAPPING = {
    'pants': 'bottom', 'cargo pants': 'bottom', 'jeans': 'bottom', 'shorts': 'bottom',
    't-shirts-short-sleeve': 'top', 't-shirts-long-sleeve': 'top', 'shirts': 'top',
    'jumpers': 'outer', 'jackets': 'outer', 'dresses': 'dress',
}


def load_catalogue(path=CATALOGUE_PATH):
    products = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'product_id', 'category', 'product_display_name', 'base_color'}
    if not required.issubset(products.columns):
        raise ValueError(f'Missing catalogue fields: {sorted(required-set(products.columns))}')
    if products.empty or products['product_id'].str.strip().eq('').any() or products['product_id'].duplicated().any():
        raise ValueError('Catalogue must contain nonempty, unique product IDs')
    categories = products['category'].map(CATEGORY_MAPPING)
    if categories.isna().any():
        raise ValueError(f'Unsupported categories: {products.loc[categories.isna(), "category"].unique().tolist()}')
    # Compatibility fields exist only in memory; never write another product CSV.
    return products.assign(id=products['product_id'], mainCategory=categories,
                           productDisplayName=products['product_display_name'],
                           baseColour=products['base_color'].str.title())


def load_catalogue_and_embeddings(catalogue_path=CATALOGUE_PATH,
                                  embedding_path=EMBEDDINGS_PATH):
    products = load_catalogue(catalogue_path)
    embeddings = np.load(embedding_path, allow_pickle=False)
    if embeddings.ndim != 2 or len(products) != len(embeddings):
        raise ValueError('Catalogue rows and embedding rows must match; regenerate embeddings')
    return products, embeddings
