from pathlib import Path

folder = Path("query-images-segmented")

# Get PNG files sorted by their current filename
images = sorted(folder.glob("*.png"))

# Temporary rename to avoid conflicts
for i, image in enumerate(images, start=1):
    image.rename(folder / f"temp_{i}.png")

# Final rename
for i in range(1, len(images) + 1):
    (folder / f"temp_{i}.png").rename(folder / f"{i}.png")

print(f"Renamed {len(images)} images.")