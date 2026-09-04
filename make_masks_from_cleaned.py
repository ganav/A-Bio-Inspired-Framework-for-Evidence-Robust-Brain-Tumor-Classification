#!/usr/bin/env python3
"""
Create binary brain-region masks from ALREADY CLEANED brain-only MRI images.

Use this only when you already have an image version in which the non-brain
region has been intentionally removed/painted black. This utility DOES NOT
perform medical brain segmentation. It simply converts your cleaned region
into a binary mask.

Input layout should mirror the Kaggle dataset:
CLEANED_ROOT/
    Training/
        glioma/
        meningioma/
        notumor/
        pituitary/
    Testing/
        glioma/
        meningioma/
        notumor/
        pituitary/

Output:
MASK_ROOT/
    Training/...
    Testing/...

White (255) = brain region
Black (0)   = non-brain region
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASSES = ("glioma", "meningioma", "pituitary", "notumor")


def list_images(directory: Path):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def cleaned_to_mask(
    image: np.ndarray,
    threshold: int,
    open_kernel: int,
    close_kernel: int,
) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Pixels above the near-black threshold are treated as the retained region.
    support = np.where(gray > threshold, 255, 0).astype(np.uint8)

    if open_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        support = cv2.morphologyEx(support, cv2.MORPH_OPEN, k)

    if close_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, k)

    # Keep only the largest connected retained region.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        support, connectivity=8
    )
    if n <= 1:
        return support

    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = np.where(labels == label, 255, 0).astype(np.uint8)

    # Fill interior holes so the result is a solid binary brain-region ROI.
    contours, _ = cv2.findContours(
        main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    mask = np.zeros_like(main)
    cv2.drawContours(
        mask, contours, contourIdx=-1, color=255, thickness=cv2.FILLED
    )
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cleaned-root", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, required=True)
    p.add_argument("--threshold", type=int, default=10)
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--close-kernel", type=int, default=5)
    args = p.parse_args()

    total = 0
    for split in ("Training", "Testing"):
        for cls in CLASSES:
            src_dir = args.cleaned_root / split / cls
            dst_dir = args.mask_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            if not src_dir.is_dir():
                raise FileNotFoundError(src_dir)

            for src in list_images(src_dir):
                image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError(f"Could not read: {src}")

                mask = cleaned_to_mask(
                    image,
                    threshold=args.threshold,
                    open_kernel=args.open_kernel,
                    close_kernel=args.close_kernel,
                )
                dst = dst_dir / f"{src.stem}.png"
                cv2.imwrite(str(dst), mask)
                total += 1

    print(f"Created {total} binary masks under {args.mask_root}")


if __name__ == "__main__":
    main()
