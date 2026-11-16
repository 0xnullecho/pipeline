import numpy as np


def _metadata_label(value):
    """Missing or non-text metadata provides no reranking evidence."""
    return value.strip().casefold() if isinstance(value, str) else ""

def _color_label(value):
    label = _metadata_label(value)
    return {'grey': 'gray', 'navy blue': 'navy', 'maroon': 'burgundy'}.get(label, label)


def search_products(products, embeddings, query_embedding, main_category=None):
    """Retrieve up to 100 neighbours within the requested garment category.

    Omitting the category retains unrestricted search. Returned indices always
    refer to the original metadata/embedding positions, not the filtered subset.
    """
    if len(products) != len(embeddings):
        raise ValueError(
            f"Catalogue has {len(products)} products but {len(embeddings)} embeddings. "
            "Regenerate embeddings from the source catalogue in CSV order."
        )

    if main_category is None:
        eligible = np.arange(len(products))
    else:
        categories = products["mainCategory"].fillna("").astype(str).str.strip().str.casefold()
        eligible = np.flatnonzero(categories.eq(main_category.strip().casefold()).to_numpy())
    if len(eligible) == 0:
        return []

    similarities = embeddings[eligible] @ query_embedding
    sorted_indices = np.argsort(similarities)[::-1]
    
    top_100 = []
    
    for position in sorted_indices[:100]:
        index = int(eligible[position])
        top_100.append({
            "index": index,
            "similarity": similarities[position],
            "product": products.iloc[index].to_dict()
        })
    return top_100


def rerank_search_results(top_100, garment_color, main_category):
    reranked = []
    for item in top_100:
        product = item["product"]
        final_similarity = float(item["similarity"])

        category = _metadata_label(product.get("mainCategory"))
        query_category = _metadata_label(main_category)
        if category and query_category:
            final_similarity += 0.04 if category == query_category else -0.03

        # A failed segmentation provides no colour evidence, positive or negative.
        color = _color_label(product.get("base_color")) or _color_label(product.get("baseColour"))
        query_color = _color_label(garment_color)
        if color and query_color:
            final_similarity += 0.04 if color == query_color else -0.03
        
        reranked.append({**item, "final_similarity": final_similarity})
    
    reranked.sort(key=lambda item: item["final_similarity"], reverse=True)
    
    return reranked
