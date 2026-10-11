import torch
from PIL import Image



def detect_and_crop_garment(
    image,
    detector_processor,
    device,
    detector,
    garment_label,
    threshold=0.3,
    padding=0.05,
):
    
    # Grounding DINO accepts textual object descriptions.
    text_labels = [[garment_label]]

    inputs = detector_processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = detector(**inputs)

    result = detector_processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[(image.height, image.width)],
        text_labels=text_labels,
    )[0]

    if len(result["boxes"]) == 0:
        # Detection failed, so use the original image.
        print("Detection failed")
        return image

    # Select the highest-confidence detection.
    best_index = result["scores"].argmax()
    box = result["boxes"][best_index].tolist()

    left, top, right, bottom = box

    # Add padding around the detected garment.
    box_width = right - left
    box_height = bottom - top

    horizontal_padding = box_width * padding
    vertical_padding = box_height * padding

    left = max(0, int(left - horizontal_padding))
    top = max(0, int(top - vertical_padding))
    right = min(image.width, int(right + horizontal_padding))
    bottom = min(image.height, int(bottom + vertical_padding))

    return image.crop((left, top, right, bottom))