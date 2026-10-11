import numpy as np

def search_products(products, embeddings, query_embedding):
    
    similarities = embeddings @ query_embedding
    
    sorted_indices = np.argsort(similarities)[::-1]
    
    top_100 = []
    
    for index in sorted_indices[:100]:
        top_100.append({
            "index": index,
            "similarity": similarities[index],
            "product": products.iloc[index].to_dict()
        })
    return top_100


def rerank_search_results(top_20, base_color, subCategory, gender, articleType):
    reranked = []
    for item in top_20:
        product = item["product"]
        final_similarity = float(item["similarity"])
        
        if product["articleType"] == articleType:
            final_similarity += 0.035

        if product["baseColour"] == base_color:
            final_similarity += 0.025

        if product["subCategory"] == subCategory:
            final_similarity += 0.010

        if product["gender"] == gender:
            final_similarity += 0.005
        
        reranked.append({**item, "final_similarity": final_similarity})
    
    reranked.sort(key=lambda item: item["final_similarity"], reverse=True)
    
    return reranked