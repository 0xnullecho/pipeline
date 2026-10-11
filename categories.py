category_prompts = {
    "Topwear": "a photo of a shirt",
    "Bottomwear": "a photo of pants",
    "Dress": "a photo of a dress",
    "Shoes": "a photo of shoes",
    "Bags": "a photo of a bag",
    "Belts": "a photo of a belt",
    "Jewellery": "a photo of jewelry",
    "Eyewear": "a photo of glasses",
    "Headwear": "a photo of a hat",
    "Scarves": "a photo of a scarf",
    "Watches": "a photo of a watch",
    "Wallets": "a photo of a wallet",
    "Innerwear": "a photo of underwear",
    "Loungewear and Nightwear": "a photo of pajamas",
    "Socks": "a photo of socks"
}

categories = list(category_prompts.keys())
texts = list(category_prompts.values())

from PIL import Image
import torch
import numpy as np
import pandas as pd


def embed_text(model, processor):
    text_inputs = processor(
        text=texts,
        padding="max_length",
        return_tensors="pt"
    )
    
    with torch.no_grad():
        output = model.get_text_features(**text_inputs)

    text_embedding = output.pooler_output

    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    text_embedding = text_embedding.cpu().numpy()
    
    return text_embedding

    

def predict_category(model, processor, query_embedding):
    text_embedding = embed_text(model, processor)
    
    similarities = text_embedding @ query_embedding
    
    similarities_index = similarities.argsort(descending=True)
    print("Predicted categoriy:",categories[similarities_index[0]] )
    
    return categories[similarities_index[0]]
