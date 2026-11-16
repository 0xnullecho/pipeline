from functools import lru_cache
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, SegformerForSemanticSegmentation
from PIL import Image

from src.catalogue import CATALOGUE_PATH, MODEL_NAME as COLOR_MODEL_NAME

MODEL_NAME = "mattmdjaga/segformer_b2_clothes"
device = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORY_TO_SEGMENTATION = {
    "top": ["Upper-clothes"],
    "bottom": ["Pants", "Skirt"],
    "dress": ["Dress"],
    "outer": ["Upper-clothes"],
    "shoes": ["Left-shoe", "Right-shoe"],
    "bag": ["Bag"],
}


@lru_cache(maxsize=1)
def _segmentation_model():
    # Load segmentation only when a query needs it.
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device).eval()
    return processor, model


def segment_clothing(image, main_category):
    category = str(main_category).strip().lower()
    if category not in CATEGORY_TO_SEGMENTATION:
        raise ValueError(f"Unknown main category: {main_category}")
    processor, model = _segmentation_model()
    clothing_labels = {label.lower() for label in CATEGORY_TO_SEGMENTATION[category]}
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = F.interpolate(outputs.logits, size=(image.height, image.width),
                               mode="bilinear", align_corners=False)
        segmentation = logits.argmax(dim=1)[0].cpu().numpy()
    label_ids = [int(idx) for idx, label in model.config.id2label.items()
                 if label.lower() in clothing_labels]
    return np.isin(segmentation, label_ids)


# Labels remain in the catalogue's existing vocabulary. Fine shades map back to
# a searchable base colour; taupe is not a separate colour in this catalogue.
COLOR_ALIASES = {
    'taupe': 'brown', 'grey-brown': 'brown', 'coffee brown': 'brown', 'tan': 'brown',
    'charcoal': 'gray', 'grey': 'gray', 'off white': 'white',
    'navy blue': 'navy', 'maroon': 'burgundy', 'turquoise blue': 'teal',
    'sea green': 'green', 'lime green': 'green', 'mustard': 'yellow',
    'rust': 'orange', 'magenta': 'pink', 'peach': 'pink',
    'mauve': 'purple', 'gold': 'yellow',
}


@lru_cache(maxsize=4)
def _read_catalogue_colors(path, modified_ns, size):
    # File metadata invalidates the cache when the catalogue changes.
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if 'base_color' not in (reader.fieldnames or []):
            raise ValueError(f'{path}: missing base_color column')
        labels = {row['base_color'].strip().casefold() for row in reader if row['base_color'].strip()}
    if not labels:
        raise ValueError(f'{path}: catalogue contains no colour labels')
    return tuple(sorted(labels))


def catalogue_color_labels(path=None):
    path = Path(path or CATALOGUE_PATH).resolve()
    stat = path.stat()
    return _read_catalogue_colors(str(path), stat.st_mtime_ns, stat.st_size)


GARMENT_DESCRIPTIONS = {
    'top': 'top', 'bottom': 'trousers or skirt', 'dress': 'dress',
    'outer': 'jacket or sweater', 'shoes': 'shoe', 'bag': 'bag',
}
# A cosine-similarity gap, not a calibrated probability. Ambiguous colour
# predictions should contribute no reranking evidence rather than a wrong bonus.
MIN_COLOR_MARGIN = 0.001


@lru_cache(maxsize=1)
def _color_model():
    processor = AutoProcessor.from_pretrained(COLOR_MODEL_NAME, use_fast=False)
    model = AutoModel.from_pretrained(COLOR_MODEL_NAME).to(device).eval()
    return processor, model


def _feature_tensor(features):
    return features.pooler_output if hasattr(features, 'pooler_output') else features


@lru_cache(maxsize=len(GARMENT_DESCRIPTIONS))
def _color_text_features(main_category, canonical):
    processor, model = _color_model()
    labels = canonical + tuple(alias for alias, target in COLOR_ALIASES.items()
                               if target in canonical and alias not in canonical)
    garment = GARMENT_DESCRIPTIONS[main_category]
    prompts = [f'A photo of a {label} {garment}. The main fabric color is {label}.'
               for label in labels]
    inputs = processor(text=prompts, padding='max_length', truncation=True,
                       max_length=model.config.text_config.max_position_embeddings,
                       return_tensors='pt').to(device)
    with torch.inference_mode():
        features = _feature_tensor(model.get_text_features(**inputs))
    return labels, F.normalize(features.float(), dim=-1)


def isolate_garment(image, mask):
    """Tight, square crop with non-garment pixels replaced by neutral grey.

    Preserves fabric colours and aspect ratio; excludes skin, background and
    other garments before the zero-shot classifier sees the image.
    """
    pixels = np.asarray(image.convert('RGB'))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != pixels.shape[:2]:
        raise ValueError('Garment mask dimensions must match the image')
    if not mask.any():
        raise ValueError('No clothing pixels detected')
    ys, xs = np.nonzero(mask)
    left, right, top, bottom = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    isolated = np.full_like(pixels[top:bottom, left:right], 224)
    selected = mask[top:bottom, left:right]
    isolated[selected] = pixels[top:bottom, left:right][selected]
    garment = Image.fromarray(isolated)
    side = max(garment.size)
    # Small border prevents the garment touching the processor's image boundary.
    border = max(1, round(side * 0.03))
    canvas = Image.new('RGB', (side+2*border, side+2*border), (224, 224, 224))
    canvas.paste(garment, ((canvas.width-garment.width)//2,
                          (canvas.height-garment.height)//2))
    return canvas


def classify_garment_colors(garment_image, main_category):
    """Rank catalogue colours by SigLIP2 image/text cosine similarity.

    Scores are relative similarities, not confidence probabilities. Text vectors
    and the model are cached, so only image encoding repeats for each garment.
    """
    category = str(main_category).strip().lower()
    if category not in GARMENT_DESCRIPTIONS:
        raise ValueError(f'Unknown main category: {main_category}')
    processor, model = _color_model()
    canonical = catalogue_color_labels()
    labels, text_features = _color_text_features(category, canonical)
    inputs = processor(images=garment_image, return_tensors='pt').to(device)
    with torch.inference_mode():
        image_features = F.normalize(_feature_tensor(model.get_image_features(**inputs)).float(), dim=-1)
        scores = (image_features @ text_features.T)[0].cpu().tolist()
    if not np.isfinite(scores).all():
        raise ValueError('Colour model returned non-finite scores')
    grouped = {}
    for label, score in zip(labels, scores):
        target = label if label in canonical else COLOR_ALIASES.get(label)
        if target in canonical:
            grouped[target] = max(grouped.get(target, -float('inf')), score)
    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def analyze_image(rgb_image, main_category, *, min_margin=MIN_COLOR_MARGIN,
                  allow_crop_fallback=False):
    """Return a zero-shot catalogue colour, or None for no mask/ambiguous colour.

    Set allow_crop_fallback only for an already detected garment bounding box;
    a missing segmentation then uses that crop directly. Arbitrary full photos
    still abstain on empty masks. Model-loading errors are surfaced to the caller.
    """
    if not np.isfinite(min_margin) or min_margin < 0:
        raise ValueError('min_margin must be a finite nonnegative similarity gap')
    mask = segment_clothing(rgb_image, main_category)
    if mask.any():
        garment = isolate_garment(rgb_image, mask)
    elif allow_crop_fallback:
        garment = rgb_image.convert('RGB')
        print(f'{main_category}: empty segmentation; classifying detector crop directly')
    else:
        print(f'{main_category}: colour unavailable because garment segmentation is empty')
        return None
    ranked = classify_garment_colors(garment, main_category)
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1]-ranked[1][1] < min_margin:
        print(f'{main_category}: ambiguous colour: {ranked[0][0]} vs {ranked[1][0]} '
              f'(similarity gap {ranked[0][1]-ranked[1][1]:.5f}, required {min_margin:.5f})')
        return None
    return ranked[0][0]
