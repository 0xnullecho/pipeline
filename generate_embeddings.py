fashion_categories = [
    "Topwear",
    "Bottomwear",
    "Dress",
    "Shoes",
    "Bags",
    "Belts",
    "Jewellery",
    "Eyewear",
    "Headwear",
    "Scarves",
    "Watches",
    "Wallets",
    "Innerwear",
    "Loungewear and Nightwear",
    "Socks"
]

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from detection import detect_and_crop_garment
from query import load_query_image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

MODEL_NAME = "google/siglip2-base-patch16-224"
BATCH_SIZE = 32
DETECTION_MODEL = "IDEA-Research/grounding-dino-tiny"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

detector_processor = AutoProcessor.from_pretrained(DETECTION_MODEL)
detector = AutoModelForZeroShotObjectDetection.from_pretrained(
    DETECTION_MODEL
).to(device)

detector.eval()

image_dir = Path("./data/images")
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

styles = pd.read_csv("./data/styles.csv", on_bad_lines="skip")
styles = styles[
    styles["subCategory"].isin(fashion_categories)
].reset_index(drop=True)

model = AutoModel.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# First determine which products have an image.
valid_products = []
valid_paths = []

for row in styles.itertuples(index=False):
    path = image_dir / f"{row.id}.jpg"

    if path.exists():
        valid_products.append(row._asdict())
        valid_paths.append(path)

valid_paths = valid_paths[:750]
valid_products = valid_products[:750]

embedding_dimension = model.config.vision_config.hidden_size

# Data is written directly to disk instead of accumulated in RAM.
embedding_file = output_dir / "cropped_embeddings.npy"
embeddings = np.lib.format.open_memmap(
    embedding_file,
    mode="w+",
    dtype=np.float32,
    shape=(len(valid_paths), embedding_dimension),
)

for start in range(0, len(valid_paths), BATCH_SIZE):
    paths = valid_paths[start : start + BATCH_SIZE]
    product_batch = valid_products[start : start + BATCH_SIZE]

    images = []

    for path, product in zip(paths, product_batch):
        
        rgb_image = load_query_image(path)
        
        cropped_image = detect_and_crop_garment(rgb_image, detector_processor, device, detector, product["subCategory"] )
        print("Cropped image:", cropped_image)
        images.append(rgb_image)
        
        
    inputs = processor(images=images, return_tensors="pt")
    
    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.inference_mode():
        output = model.get_image_features(**inputs)
        batch_embeddings = output.pooler_output
        batch_embeddings = torch.nn.functional.normalize(
            batch_embeddings,
            dim=-1,
        )

    end = start + len(paths)
    embeddings[start:end] = batch_embeddings.cpu().numpy()
    embeddings.flush()

    print(f"Embedded {end}/{len(valid_paths)}")

pd.DataFrame(valid_products).to_csv(
    output_dir / "cropped_products.csv",
    index=False,
)