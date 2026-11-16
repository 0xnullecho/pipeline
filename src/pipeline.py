from src.retrieval.search import search_products, rerank_search_results
from src.embeddings.encoder import load_query_image, embed_query_image
import torch
from src.detection.color_detection import analyze_image, CATEGORY_TO_SEGMENTATION
from transformers import (
    AutoModel, AutoImageProcessor, AutoModelForObjectDetection
)
from src.detection.detection import detect_garment_pieces
import argparse
from pathlib import Path
from types import SimpleNamespace
from src.catalogue import load_catalogue_and_embeddings

MODEL_NAME = "google/siglip2-base-patch16-224"
IMAGE_MODEL_NAME = 'yainage90/fashion-object-detection'

def load_pipeline():
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    image_model = AutoModelForObjectDetection.from_pretrained(
        IMAGE_MODEL_NAME,
        low_cpu_mem_usage=False,
    ).to(device)

    image_model.eval()


    fashion_df, embeddings = load_catalogue_and_embeddings()
    return SimpleNamespace(device=device, processor=processor, model=model, image_processor=image_processor, image_model=image_model, fashion_df=fashion_df, embeddings=embeddings)


def run_query(image_path, pipeline):
    device = pipeline.device
    processor = pipeline.processor
    model = pipeline.model
    image_processor = pipeline.image_processor
    image_model = pipeline.image_model
    fashion_df = pipeline.fashion_df
    embeddings = pipeline.embeddings
    # load image and convert to rgb
    rgb_image = load_query_image(image_path)

    # detect different clothing items, isolate, crop and returns them in list with related MainCategory for later reranking
    image_bounding_boxes = detect_garment_pieces(rgb_image, image_processor, image_model, device)

    item_results = []
    for score, label_id, coordinates in image_bounding_boxes:
        label = image_model.config.id2label[label_id]
        if label.strip().lower() not in CATEGORY_TO_SEGMENTATION:
            print(f"{label}: unsupported garment category; skipping detection")
            continue

        # crop target image by bounding box
        cropped_image = rgb_image.crop(coordinates)

        # get base color for query clothing
        base_color = analyze_image(cropped_image, label, allow_crop_fallback=True) 
        if base_color is None:
            print(f"{label}: no reliable garment colour; skipping colour reranking")

        # create image embedding
        query_embedding = embed_query_image(
            model,
            processor,
            cropped_image,
            device
        )

        # compare embedding to existing product catalogue embeddings
        top_products = search_products(
            fashion_df,
            embeddings,
            query_embedding,
            main_category=label,
        )

        # rerank the results based on metadata attributes
        reranked = rerank_search_results(
            top_products,
            base_color,
            label
        )
        item_results.append((label, {"query_base_color": base_color, "products": reranked}))
    return item_results


def write_results(image_path, item_results, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not item_results:
        with open(output_path, "a") as file:
            file.write(f"No garments detected for: {image_path} (no supported detections)\n\n")
        return
    for key, value in item_results:
        with open(output_path, "a") as file:
            file.write(
                f"Top 5 for: {image_path}. \nLabel: {key} \nQuery base color: {value['query_base_color']}\n"
            )

            for item in value["products"][:5]:
                product = item["product"]
                file.write(f"Product ID: {product["id"]} \nBase colour: {product["baseColour"]} \nMainCategory: {product["mainCategory"]} \nInitial similarity index: {item["similarity"]} \nFinal similarity: {item["final_similarity"]} \n\n")

            file.write("\n\n\n")


def main():
    parser = argparse.ArgumentParser(description="Find similar clothing items")
    parser.add_argument("--photo", required=True, help="Path to query photo")
    parser.add_argument("--output", default="./results/manual-query-results.txt", help="Append results to this file")
    args = parser.parse_args()
    pipeline = load_pipeline()
    write_results(args.photo, run_query(args.photo, pipeline), args.output)


if __name__ == "__main__":
    main()
