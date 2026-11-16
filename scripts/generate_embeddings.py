"""Encode catalogue images in CSV order and save their normalized vectors."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from scripts.ImageDataset import ProductImageDataset
from src.catalogue import (
    EMBEDDINGS_PATH, MODEL_NAME, ROOT, load_catalogue,
)

BATCH_SIZE = 32


def main():
    products = load_catalogue()
    image_dir = ROOT / 'data/synthetic-catalogue/images'
    paths = [image_dir / f'{product_id}.jpg' for product_id in products['product_id']]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images missing; first: {missing[:5]}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    loader = DataLoader(ProductImageDataset(paths, processor),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    batches = []
    with torch.inference_mode():
        for pixels in tqdm(loader, desc='Generating embeddings'):
            features = model.get_image_features(pixel_values=pixels.to(device))
            if hasattr(features, 'pooler_output'):
                features = features.pooler_output
            batches.append(F.normalize(features.float(), dim=-1).cpu().numpy())
    vectors = np.concatenate(batches)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, vectors)
    print(f'Saved {len(vectors)} embeddings in catalogue order.')


if __name__ == '__main__':
    main()
