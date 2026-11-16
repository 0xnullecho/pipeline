from PIL import Image
from torch.utils.data import Dataset


class ProductImageDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = image.convert("RGB")

            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"][0]

        return pixel_values