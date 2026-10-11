from transformers import AutoModel, AutoProcessor
import pandas as pd
import os
import argparse
from search import search_products, rerank_search_results
from query import load_query_image, embed_query_image
from categories import predict_category
import numpy as np
import torch
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,AutoImageProcessor, AutoModelForObjectDetection
)
from PIL import Image
from pathlib import Path
from test_cases import test_cases
from detection import detect_garment_pieces

testing_queries = os.listdir("./query-images-segmented")

MODEL_NAME = "google/siglip2-base-patch16-224"
IMAGE_MODEL_NAME = 'yainage90/fashion-object-detection'

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    
    
model = AutoModel.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
image_model = AutoModelForObjectDetection.from_pretrained(IMAGE_MODEL_NAME).to(device)



products = pd.read_csv("./output/original_products.csv", sep=",")
products = pd.DataFrame(products)
embeddings = np.load("./output/embeddings.npy")


for test_case in test_cases[:1]:
    path = test_case["filename"]

# load image and convert to rgb (3 channels)
    rgb_image = load_query_image(
        f"./query-images-segmented/{path}"
    )

# crop the image with bounding box to remove noise from image (background, etc)
    cropped_images = detect_garment_pieces(rgb_image, image_processor, image_model, device)

# Select first crop for testing purposes otherwise all will be sent to frotnend for user selection
    image = cropped_images[0]["cropped_image"]

        
# create image embedding
    query_embedding = embed_query_image(
        model,
        processor,
        image,
    )

# compare embedding to existing product catalogue embeddings
    top_products = search_products(
        products,
        embeddings,
        query_embedding,
    )

# rerank the results based on metadata attributes
reranked = rerank_search_results(
    top_products,
    test_case["baseColour"],
    test_case["category"],
    test_case["gender"],
    test_case["articleType"]
)

print(reranked)
with open("./results/new-test.txt", "a") as file:
    file.write(
        f"Segmented results for: {path} -> {test_case}\n"
    )

    for result in reranked[:10]:
        file.write(f"Product: {result}\n")

    file.write("\n\n\n")


# argparser = argparse.ArgumentParser(prog="Find cheaper, alternative fashion items by providing an image.")
# argparser.add_argument("--photo", required=True, help="Path to photo")

# argparser.add_argument("--base-color", required=True, help="Specify clothing items base color (e.g black)")
# argparser.add_argument("--category", required=True, choices=["Topwear", "Bottomwear", "Dress", "Shoes", "Bags", "Belts", "Jewellery", "Eyewear", "Headwear", "Scarves", "Watches", "Wallets", "Innerwear", "Loungewear and Nightwear", "Socks"], help="Type of clothing item")
# argparser.add_argument("--gender", required=False, choices=["Girls", "Men", "Unisex"], help="Gender for which the clothing item is meant.", default="Unisex")
# argparser.add_argument("--garment-label", required=True, help="Specify the type of target clothing to detect and extract.")

# args = argparser.parse_args()

# path_to_query_image = args.photo
# base_color = args.base_color
# category = args.category
# gender = args.gender
# garment_label = args.garment_label

# DETECTION_MODEL = "IDEA-Research/grounding-dino-tiny"

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# detector_processor = AutoProcessor.from_pretrained(DETECTION_MODEL)
# detector = AutoModelForZeroShotObjectDetection.from_pretrained(
#     DETECTION_MODEL
# ).to(device)

# detector.eval()


# query_image = load_query_image(path_to_query_image)

# # with Image.open() as image:
# cropped_image = detect_and_crop_garment(query_image, detector_processor, device, detector, garment_label)

# cropped_image.save("cropped_query5.jpg")


# # print(cropped_image)

# # query embedding
# query_embedding = embed_query_image(model, processor, query_image)

# # predict category based on query_embedding
# # category_prediction = predict_category(model, processor, query_embedding)

# top_products = search_products(products, embeddings, query_embedding)

# reranked = rerank_search_results(top_products, base_color, category, gender)

# for product in reranked[:5]:
#     print("Product:", product["final_similarity"], product["product"]["id"],  "\n")
    
# with open("./output/no-crop-query_image.txt", "w+") as file:
#     file.write(f"Input:, {path_to_query_image}, {base_color}, {category}, {gender} \n")
#     file.write("Products:")
#     for product in reranked[:5]:
#         file.write(f"{product} \n")
#     file.write("\n\n")