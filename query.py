import torch 
import os
from PIL import Image


def load_query_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")
    
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
    
    return rgb_image


def embed_query_image(model, processor, query_image):

    query_input = processor(
        images=[query_image],
        return_tensors="pt"
    )
    
    with torch.inference_mode():
        output = model.get_image_features(**query_input)

    embedding = output.pooler_output

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    embedding = embedding.cpu().numpy()[0]

    return embedding
