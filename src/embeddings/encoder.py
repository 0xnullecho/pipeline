import torch
from pathlib import Path
from PIL import Image


def load_query_image(image_path):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image path not found: {image_path}")
    
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
    
    return rgb_image


def embed_query_image(model, processor, query_image, device):

    query_input = processor(
        images=[query_image],
        return_tensors="pt"
    ).to(device)
    
    with torch.inference_mode():
        output = model.get_image_features(**query_input)

    embedding = output

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0]
