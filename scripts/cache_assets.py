"""Pre-download and cache all assets needed for training and evaluation."""

import os

import open_clip
from datasets import load_dataset
from huggingface_hub import hf_hub_download

cache_dir = os.path.join(os.environ.get("BASE_DIR", "."), "model-merging", "checkpoints", "openclip_cache")
os.makedirs(cache_dir, exist_ok=True)

print(f"  Downloading OpenCLIP ViT-B-16 (openai) to {cache_dir}...")
open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", cache_dir=cache_dir)
print("  Done.")

# Cache the zeroshot encoder from HuggingFace (used by load_model_from_hf)
print("  Downloading crisostomi/ViT-B-16-base from HuggingFace Hub...")
hf_hub_download(repo_id="crisostomi/ViT-B-16-base", filename="pytorch_model.bin")
print("  Done.")

# Cache MNIST dataset
print("  Downloading MNIST dataset...")
load_dataset("ylecun/mnist")
print("  Done.")

# Cache EuroSAT dataset
print("  Downloading EuroSAT dataset...")
load_dataset("tanganke/eurosat")
print("  Done.")

print("\nAll assets cached successfully.")
