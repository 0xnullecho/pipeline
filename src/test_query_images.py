"""Run from the repository root: python -m src.test_query_images."""

from pathlib import Path

from src.pipeline import load_pipeline, run_query, write_results

QUERY_SETS = (
    ("query-images", "results/results-original.txt"),
    ("query-images-segmented", "results/results-segmented.txt"),
    ("query-images-cropped", "results/results-cropped.txt"),
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    # Load the models and catalogue once for every image across all three sets.
    pipeline = None
    failures = 0
    for directory, output_path in QUERY_SETS:
        folder = Path(directory)
        if not folder.is_dir():
            print(f"Skipping missing folder: {folder}")
            continue
        images = sorted(path for path in folder.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            print(f"No images in: {folder}")
            continue
        if pipeline is None:
            pipeline = load_pipeline()
        for index, image_path in enumerate(images, start=1):
            print(f"[{directory} {index}/{len(images)}] {image_path.name}")
            try:
                results = run_query(str(image_path), pipeline)
            except Exception as error:
                failures += 1
                message = f"ERROR for {image_path}: {type(error).__name__}: {error}"
                print(message)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "a") as file:
                    file.write(message + "\n\n")
                continue
            write_results(str(image_path), results, output_path)
        print(f"Results appended to {output_path}")
    if failures:
        raise SystemExit(f"Batch finished with {failures} failed image(s); see results files.")


if __name__ == "__main__":
    main()
