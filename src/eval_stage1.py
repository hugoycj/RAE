#!/usr/bin/env python3
"""
Evaluate Stage-1 RAE reconstruction: compute PSNR and save input/output images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path: Path, target_size: int | None = None) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if target_size is not None:
        image = image.resize((target_size, target_size), Image.BICUBIC)
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W)
    return tensor


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute PSNR between two images (values in [0, 1])."""
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return float('inf')
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-1 RAE: compute PSNR and save input/output images."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config with a stage_1 section.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output"),
        help="Directory to save output images (default: eval_output).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for evaluation (default: 256).",
    )
    parser.add_argument(
        "--device",
        help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    # Parse config and instantiate model
    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    # Add checkpoint path to config
    rae_config = dict(rae_config)
    rae_config["ckpt"] = args.ckpt

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_save_dir = args.output_dir / "input"
    output_save_dir = args.output_dir / "output"
    comparison_dir = args.output_dir / "comparison"
    input_save_dir.mkdir(parents=True, exist_ok=True)
    output_save_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in args.input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise ValueError(f"No images found in {args.input_dir}")

    print(f"Found {len(image_files)} images to evaluate")
    print(f"Using device: {device}")
    print(f"Image size: {args.image_size}")

    psnr_values = []

    for idx, image_path in enumerate(image_files):
        # Load image
        image = load_image(image_path, args.image_size).to(device)

        # Reconstruct
        with torch.no_grad():
            latent = rae.encode(image)
            recon = rae.decode(latent)

        # Clamp reconstruction to valid range
        recon = recon.clamp(0.0, 1.0)

        # Compute PSNR
        psnr = compute_psnr(image, recon)
        psnr_values.append(psnr)

        # Save images
        stem = image_path.stem
        save_image(image, input_save_dir / f"{stem}.png")
        save_image(recon, output_save_dir / f"{stem}.png")

        # Save side-by-side comparison
        comparison = torch.cat([image, recon], dim=3)  # Concatenate horizontally
        save_image(comparison, comparison_dir / f"{stem}_comparison.png")

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(image_files)}] {stem}: PSNR = {psnr:.2f} dB")

    # Compute statistics
    psnr_array = np.array(psnr_values)
    mean_psnr = np.mean(psnr_array)
    std_psnr = np.std(psnr_array)
    min_psnr = np.min(psnr_array)
    max_psnr = np.max(psnr_array)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Number of images: {len(psnr_values)}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"Std PSNR:  {std_psnr:.2f} dB")
    print(f"Min PSNR:  {min_psnr:.2f} dB")
    print(f"Max PSNR:  {max_psnr:.2f} dB")
    print("=" * 50)
    print(f"\nInputs saved to: {input_save_dir}")
    print(f"Outputs saved to: {output_save_dir}")
    print(f"Comparisons saved to: {comparison_dir}")

    # Save results to file
    results_file = args.output_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Image size: {args.image_size}\n")
        f.write(f"Number of images: {len(psnr_values)}\n")
        f.write(f"Mean PSNR: {mean_psnr:.2f} dB\n")
        f.write(f"Std PSNR:  {std_psnr:.2f} dB\n")
        f.write(f"Min PSNR:  {min_psnr:.2f} dB\n")
        f.write(f"Max PSNR:  {max_psnr:.2f} dB\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-image PSNR:\n")
        for image_path, psnr in zip(image_files, psnr_values):
            f.write(f"{image_path.name}: {psnr:.2f} dB\n")

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
