# Project setup

Run commands from the repository root. The project uses local catalogue files and downloads pretrained models on first use, so the initial run requires network access and disk space for model weights.

## 1. Prepare a Python environment

Use **Python 3.12**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it in PowerShell with `.venv\Scripts\Activate.ps1`.

Install **one** PyTorch build before installing the remaining requirements:

- **CPU (Linux/Windows):** no GPU driver is needed. This is the simplest setup, although inference and catalogue embedding generation will be slower.

  ```bash
  python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
  ```

- **NVIDIA GPU (Linux/Windows):** install a compatible NVIDIA driver, then choose a PyTorch 2.5.1 CUDA build for your system using the [official previous-version installation commands](https://pytorch.org/get-started/previous-versions/). For example, the CUDA 12.1 build:

  ```bash
  python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  ```

- **AMD GPU (Linux):** check GPU/OS support and driver requirements in the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). The local test environment uses this ROCm 6.2 build:

  ```bash
  python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/rocm6.2
  ```

- **macOS:** install `torch==2.5.1 torchvision==0.20.1` from the default package index. The query pipeline selects Apple MPS when available; catalogue embedding generation, color classification and clothing segmentation currently use CPU on macOS. macOS has not been validated for this project.

Then install the project dependencies and check the environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('ROCm:', torch.version.hip); print('GPU available:', torch.cuda.is_available()); print('MPS available:', torch.backends.mps.is_available())"
```

ROCm uses the `torch.cuda` API too: `GPU available: True` is expected on a supported AMD setup. If it is false on an intended GPU installation, check the selected wheel, driver, and GPU compatibility before generating the catalogue. Catalogue embedding generation uses CUDA/ROCm when available, otherwise CPU. On Linux, `CUDA_VISIBLE_DEVICES="" python -m src.pipeline --photo /path/to/outfit.jpg` hides NVIDIA GPUs to force a CPU query; AMD visibility is controlled by the ROCm runtime.

`requirements.txt` pins direct dependencies from the Python 3.12 / PyTorch 2.5.1+rocm6.2 environment where the unit tests pass. It is not a complete transitive lockfile. The offline regression suite and a cached-model sample query have been run locally on CPU. A clean installation and real inference on NVIDIA and macOS have not been validated. Unit tests mock model downloads/inference. No additional test framework is required.

Models download from Hugging Face on first use. Allow network access and several GB of free space for the catalogue, model cache, and generated vectors. Models use the default Hugging Face cache; optionally set `HF_HOME` to a larger disk before running. The query uses SigLIP 2, the fashion detector, and a clothing segmentation model; catalogue embedding generation only needs SigLIP 2.

### Local AMD architecture compatibility

On the development machine, the GPU reports `gfx1031`, while the installed PyTorch 2.5.1 / ROCm 6.2 rocBLAS libraries contain `gfx1030` kernels. Without an override, inference aborts with a missing Tensile library error even though `torch.cuda.is_available()` is true.

For this tested local configuration, use a per-process override:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m src.test_query_images
```

The same prefix can be used with the single-photo query command in the walkthrough. A GPU matrix multiplication and an end-to-end query were verified with this setting. This is a hardware-specific compatibility workaround, not a requirement for other AMD GPUs or NVIDIA systems. It does not modify drivers or model files.

## 2. Prepare the synthetic catalogue

The active catalogue contains 30,000 synthetic garment specifications across ten categories. Product images are generated from their prompts; these are illustrative concepts, not verified photos of real inventory. The full product image collection, generation metadata, vectors, backups and local evaluation photos are excluded.

Use `data/synthetic-catalogue/product_catalogue_30000.csv` as the source catalogue. Supply the product JPEGs separately, named by the original product IDs:

```text
data/
├── synthetic-catalogue/
│   ├── product_catalogue_30000.csv
│   └── images/
│       ├── SYN-000001.jpg
│       └── ...
└── embeddings/
    └── product_embeddings.npy
```

Generate embeddings directly from the source catalogue, in CSV row order:

```bash
python -m scripts.generate_embeddings
```

The source CSV is the only product metadata file. The pipeline validates required fields, unique IDs and supported categories, and derives broad categories and display aliases in memory. It does not write processed or embedding-specific product CSVs.

Embedding generation requires every catalogue image to be present; it stops before loading the model if any image is missing. Vectors are written in source CSV row order. Query loading checks that the vector row count matches the catalogue. Keep the CSV in the same order used to generate the vectors; regenerate embeddings after changing or reordering it. The generator collects normalized batches in memory and saves one NumPy array after inference finishes. The final array is about 92 MB for this catalogue, with additional memory needed during concatenation. Saving overwrites the existing array; keep a copy if you need the previous version.

The query classifier reads its complete color vocabulary from the source CSV. Keep that file available alongside the generated embeddings. The current images are 512 × 512 JPEG quality 90; SigLIP2 applies its own configured 224 × 224 input processing.

## Models loaded during use

| Purpose | Model |
| --- | --- |
| Image embeddings and zero-shot color classification | `google/siglip2-base-patch16-224` |
| Garment detection | `yainage90/fashion-object-detection` |
| Clothing segmentation | `mattmdjaga/segformer_b2_clothes` |

The models are pretrained; no model training is required for setup. 

## Next: run a query

Continue to the [sample-query walkthrough](walkthrough.MD) for the command, recorded top-five matches and result thumbnails. For the project goal, scope and roadmap, see the [README](README.md).
