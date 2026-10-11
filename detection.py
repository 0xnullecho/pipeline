from PIL import Image
import torch
from transformers import  AutoImageProcessor, AutoModelForObjectDetection

# device = 'cpu'
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')

# ckpt = 'yainage90/fashion-object-detection'
# image_processor = AutoImageProcessor.from_pretrained(ckpt)
# model = AutoModelForObjectDetection.from_pretrained(ckpt).to(device)

# image = Image.open('./query-images/46.png').convert('RGB')

# with torch.no_grad():
#     inputs = image_processor(images=[image], return_tensors="pt")
#     print(inputs)
#     outputs = model(**inputs.to(device))
#     target_sizes = torch.tensor([[image.size[1], image.size[0]]])
#     results = image_processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]

#     items = []
#     for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
#         score = score.item()
#         label = label.item()
#         box = [i.item() for i in box]
        
#         print(f"{model.config.id2label[label]}: {round(score, 3)} at {box}")
#         items.append((score, label, box))
        
#         box = (box[0],box[1],box[2],box[3])
#         cropped_img = image.crop(box)
#         cropped_img.save(f"test-{idx}.png")


# Returns bounding boxes for every clothing item
def detect_garment_pieces(rgb_img, processor, model, device):
    with torch.no_grad():
        inputs = processor(images=[rgb_img], return_tensors="pt")
        
    outputs = model(pixel_values=inputs["pixel_values"].to(device), pixel_mask=inputs["pixel_mask"].to(device))
    
    target_sizes = torch.tensor([[rgb_img.size[1], rgb_img.size[0]]])
    
    processed_results = processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]
    # print("Post processing result:", processed_results)
    
    items = []
    for score, label, box in zip(processed_results["scores"], processed_results["labels"], processed_results["boxes"]):
        score = score.item()
        label = label.item()
        box = [i.item() for i in box]
        items.append((score, label, box))
        
    cropped_images = crop_load_image(items, rgb_img)
    return cropped_images

def crop_load_image(processed_results, rgb_img):
    cropped_images = []
    for score, label, box in processed_results:
        print(score, label, box)
                
        image = rgb_img.crop(box)
        cropped_images.append({
            "label": label,
            "score": score,
            "box": box,
            "cropped_image": image
        })

    return cropped_images
        