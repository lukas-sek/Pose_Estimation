"""
Preprocessing step 2: resize cropped outputs to 256x256.

Reads from ./preprocessed_data/image/ and ./preprocessed_data/depth/
(produced by preprocess.py) and writes 256x256 versions to ./resized_data/:
  - nearest neighbor  -> image_nn_256x256/, depth_nn_256x256/
  - bilinear          -> image_bilinear_256x256/, depth_bilinear_256x256/

Split .txt files in preprocessed_data/ remain valid (same sample names).
"""

import os
import time

import cv2
import numpy as np


# Configuration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(SCRIPT_DIR, 'preprocessed_data')
RESIZED_DIR = os.path.join(SCRIPT_DIR, 'resized_data')
IMAGE_DIR = os.path.join(PREPROCESSED_DIR, 'image')
DEPTH_DIR = os.path.join(PREPROCESSED_DIR, 'depth')

TARGET_SIZE = (256, 256)  # (width, height) for cv2.resize
SIZE_SUFFIX = f'{TARGET_SIZE[0]}x{TARGET_SIZE[1]}'

METHODS = {
    'nn': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
}


def fmt_duration(seconds):
    """Format a duration in seconds as HH:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def resize_split(method_name, cv2_flag):
    """
    Resize all image/depth pairs from preprocessed_data using the given method.

    Args:
        method_name: suffix for output folders ('nn' or 'bilinear')
        cv2_flag: OpenCV interpolation flag (INTER_NEAREST or INTER_LINEAR)

    Returns:
        Number of samples processed.
    """
    out_image_dir = os.path.join(
        RESIZED_DIR, f'image_{method_name}_{SIZE_SUFFIX}')
    out_depth_dir = os.path.join(
        RESIZED_DIR, f'depth_{method_name}_{SIZE_SUFFIX}')
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f'Input image directory not found: {IMAGE_DIR}\n'
            'Run preprocess.py first.'
        )

    image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg'))
    if not image_files:
        print(f'  [WARN] No .jpg files found in {IMAGE_DIR}')
        return 0

    n = len(image_files)
    start = time.time()

    for i, fname in enumerate(image_files):
        name = os.path.splitext(fname)[0]
        depth_path = os.path.join(DEPTH_DIR, name + '.npy')

        if not os.path.isfile(depth_path):
            print(f'  [WARN] Missing depth for {name}, skipping.')
            continue

        rgb = cv2.imread(os.path.join(IMAGE_DIR, fname))
        depth = np.load(depth_path).astype(np.float32)

        rgb_resized = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2_flag)
        depth_resized = cv2.resize(depth, TARGET_SIZE, interpolation=cv2_flag)

        cv2.imwrite(os.path.join(out_image_dir, fname), rgb_resized)
        np.save(os.path.join(out_depth_dir, name + '.npy'), depth_resized)

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (n - i - 1)
            print(
                f'  [{method_name}] {i + 1}/{n} | '
                f'elapsed {fmt_duration(elapsed)} | '
                f'ETA {fmt_duration(eta)}'
            )

    return n


if __name__ == '__main__':
    global_start = time.time()

    for method_name, cv2_flag in METHODS.items():
        print(f'\nResizing with {method_name} interpolation...')
        count = resize_split(method_name, cv2_flag)
        print(f'  Done: {count} samples -> '
              f'resized_data/image_{method_name}_{SIZE_SUFFIX}/, '
              f'resized_data/depth_{method_name}_{SIZE_SUFFIX}/')

    total = time.time() - global_start
    print(f'\nAll resizing complete in {fmt_duration(total)}.')
