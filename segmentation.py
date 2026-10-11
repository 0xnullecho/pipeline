import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np

# def crop_image(rgb_image):
#     pass
    
    
    
    
    

processor = SegformerImageProcessor.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)

model.eval()

img = Image.open("./query-images-cropped/1.png").convert("RGB")

inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)


logits = F.interpolate(
    outputs.logits,
    size=img.size[::-1],
    mode="bilinear",
    align_corners=False
)

segmentation = logits.argmax(dim=1)[0]
print(model.config.id2label)

garment_mask = segmentation == 4
image_np = np.array(img)
mask = garment_mask.cpu().numpy()

result = np.ones_like(image_np) * 255
result[mask] = image_np[mask]

segmented_img = Image.fromarray(result)
segmented_img.show()
plt.imshow(segmented_img)
plt.show()

from ultralytics import YOLO

# Load a pre-trained model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model("./query-images/1.png")

# Access detection results
for result in results:
    print('Result:', result, "\n\n\n")
    for box in result.boxes:
        print(f"Class: {result.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}", box)   