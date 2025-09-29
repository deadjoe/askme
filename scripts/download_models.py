#!/usr/bin/env python3
"""
Quick model download script for askme dual-model architecture.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from huggingface_hub import snapshot_download

    print("✅ Required libraries available")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)

# Configure mirror if needed
HF_MIRRORS = {
    "mirror1": "https://hf-mirror.com",
    "mirror2": "https://huggingface.com.cn",
    "official": "https://huggingface.co",
}


def download_model(model_name: str, use_mirror: str = "official") -> str | None:
    """Download model with optional mirror."""
    if use_mirror != "official":
        os.environ["HF_ENDPOINT"] = HF_MIRRORS[use_mirror]
        print(f"🌐 Using mirror: {HF_MIRRORS[use_mirror]}")

    print(f"📥 Downloading {model_name}...")

    try:
        # Download with resume capability
        cache_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=None,  # Use default cache
            resume_download=True,
            local_files_only=False,
        )
        print(f"✅ {model_name} downloaded to: {cache_dir}")
        return cache_dir
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def main() -> None:
    """Download all required models for dual-model architecture."""
    models_to_download = [
        "Qwen/Qwen3-Embedding-0.6B",
        "BAAI/bge-m3",
        "Qwen/Qwen3-Reranker-0.6B",
        "BAAI/bge-reranker-v2-m3",
    ]

    # Check available mirrors
    mirror = "official"
    if len(sys.argv) > 1:
        mirror = sys.argv[1]
        if mirror not in HF_MIRRORS:
            print(f"❌ Unknown mirror: {mirror}")
            print(f"Available: {list(HF_MIRRORS.keys())}")
            sys.exit(1)

    print(f"🚀 Starting model downloads (mirror: {mirror})")
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"🔧 PyTorch device: {device}")

    success_count = 0
    for model in models_to_download:
        print(f"\n{'='*60}")
        result = download_model(model, mirror)
        if result:
            success_count += 1
        else:
            print(f"⚠️  Failed to download {model}")

    total = len(models_to_download)
    print(f"\n🎉 Download complete: {success_count}/{total} models")

    if success_count == total:
        print("\n✅ All models ready! You can now start the API server:")
        print("   ./scripts/start-api.sh")
    else:
        print("\n⚠️  Some downloads failed. Try with mirror:")
        print("   python scripts/download_models.py mirror1")


if __name__ == "__main__":
    main()
