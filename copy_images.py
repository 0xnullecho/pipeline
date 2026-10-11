from pathlib import Path
import pandas as pd
import shutil

products = pd.read_csv("output/original_products.csv")

source = Path("data/images")
destination = Path("data/catalogue_images")

destination.mkdir(exist_ok=True)

for product_id in products["id"]:
    src = source / f"{product_id}.jpg"
    dst = destination / f"{product_id}.jpg"

    if src.exists():
        shutil.copy2(src, dst)

print(f"Copied {len(list(destination.glob('*.jpg')))} images")