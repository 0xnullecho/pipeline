import torch


def detect_garment_pieces(rgb_img, processor, model, device):
    """Return confidence, label ID and bounding box for each detected garment."""
    with torch.inference_mode():
        inputs = processor(images=[rgb_img], return_tensors="pt")
        outputs = model(pixel_values=inputs["pixel_values"].to(device),
                        pixel_mask=inputs["pixel_mask"].to(device))
    target_sizes = torch.tensor([[rgb_img.size[1], rgb_img.size[0]]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.4, target_sizes=target_sizes)[0]
    return [(score.item(), label.item(), box.tolist())
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"])]
