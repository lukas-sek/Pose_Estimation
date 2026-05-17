"""
Preprocessing script for CLOTH3D++ subset.

For each sequence folder in cloth3d++_subset/:
  - Pass 1: scan the segmentation video to determine which frames have the
    subject's mask touching any border of the image frame. These are discarded.
  - Pass 2: for every remaining frame, render the depth map, apply a square
    crop centred on the subject (10 px margin), and save the RGB frame as .jpg
    and the depth map as .npy.

Outputs go to ./preprocessed_data/image/ and ./preprocessed_data/depth/.
Split .txt files are written to ./preprocessed_data/.

Split boundaries (by sorted folder position):
  Train      : folders  1-128  (00001-00152)
  Validation : folders 129-144 (00153-00168)
  Test       : folders 145-160 (00169-00185)
"""

import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — add DataReader to sys.path so its relative imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAREADER_DIR = os.path.join(SCRIPT_DIR, 'cloth3d', 'DataReader')
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'cloth3d'))
sys.path.insert(0, DATAREADER_DIR)

from DataReader.read import DataReader          # noqa: E402
from DataReader.util import intrinsic, extrinsic  # noqa: E402
from DataReader.depth_render import Render      # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBSET_DIR  = os.path.join(SCRIPT_DIR, 'cloth3d++_subset')
OUT_DIR     = os.path.join(SCRIPT_DIR, 'preprocessed_data')
IMAGE_DIR   = os.path.join(OUT_DIR, 'image')
DEPTH_DIR   = os.path.join(OUT_DIR, 'depth')

CROP_MARGIN = 10   # extra pixels around the subject bounding box
MAX_DEPTH   = 10   # background depth threshold used in demo notebook

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DataReader — override SRC to point at the actual data location
# ---------------------------------------------------------------------------
reader = DataReader()
reader.SRC = SUBSET_DIR + os.sep


# ---------------------------------------------------------------------------
# Helper: triangulate quad faces (required by the Render class)
# ---------------------------------------------------------------------------
def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3:
            out.append(f)
        elif len(f) == 4:
            out.append([f[0], f[1], f[2]])
            out.append([f[0], f[2], f[3]])
    return np.array(out, np.int32)


# ---------------------------------------------------------------------------
# Helper: compute square crop parameters from a binary mask
# Returns (cx, cy, half) where the crop is img[cy-half:cy+half, cx-half:cx+half]
# ---------------------------------------------------------------------------
def get_square_crop(mask, margin=CROP_MARGIN):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cy = int((rmin + rmax) / 2)
    cx = int((cmin + cmax) / 2)
    half = max(rmax - rmin, cmax - cmin) // 2 + margin
    return cx, cy, half


# ---------------------------------------------------------------------------
# Helper: apply square crop (no resizing)
# ---------------------------------------------------------------------------
def crop_image(img, cx, cy, half):
    H, W = img.shape[:2]
    x1 = max(0, cx - half)
    x2 = min(W, cx + half)
    y1 = max(0, cy - half)
    y2 = min(H, cy + half)
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Helper: scan segmentation video in pass 1.
# Returns list of at_edge booleans — True if the mask touches any image border.
# An empty mask is also treated as at_edge (subject fully out of frame).
# ---------------------------------------------------------------------------
def scan_segmentation(seg_cap):
    at_edge_flags = []
    while True:
        ok, seg = seg_cap.read()
        if not ok:
            break
        mask = seg[:, :, 0] > 0
        if mask.any():
            H, W = seg.shape[:2]
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            at_edge = (rmin == 0 or rmax == H - 1 or
                       cmin == 0 or cmax == W - 1)
        else:
            at_edge = True   # empty mask → subject fully out of frame
        at_edge_flags.append(at_edge)
    return at_edge_flags


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def fmt_duration(seconds):
    """Format a duration in seconds as HH:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


# ---------------------------------------------------------------------------
# Main per-sequence processing function
# Returns list of saved frame name strings (e.g. "00001_42")
# ---------------------------------------------------------------------------
def process_sequence(folder, seq_idx, total_seqs, global_start):
    sample = folder
    saved  = []

    rgb_path  = os.path.join(SUBSET_DIR, sample, sample + '.mkv')
    segm_path = os.path.join(SUBSET_DIR, sample, sample + '_segm.mkv')

    if not os.path.isfile(rgb_path) or not os.path.isfile(segm_path):
        print(f'  [WARN] Missing video files for {sample}, skipping.')
        return saved

    rgb_cap = cv2.VideoCapture(rgb_path)
    seg_cap = cv2.VideoCapture(segm_path)

    # ------------------------------------------------------------------
    # Pass 1: scan segmentation video only to build edge-flag list
    # ------------------------------------------------------------------
    at_edge_flags = scan_segmentation(seg_cap)

    if not at_edge_flags:
        print(f'  [WARN] No frames found for {sample}.')
        rgb_cap.release()
        seg_cap.release()
        return saved

    n_total  = len(at_edge_flags)
    n_valid  = sum(1 for f in at_edge_flags if not f)

    # ------------------------------------------------------------------
    # Rewind both captures (seek, no re-open)
    # ------------------------------------------------------------------
    rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    seg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ------------------------------------------------------------------
    # Read sequence metadata once (camera params, garment list)
    # ------------------------------------------------------------------
    info     = reader.read_info(sample)
    garments = list(info['outfit'].keys())

    # Initialise renderer with camera parameters (constant per sequence)
    render = Render(max_depth=MAX_DEPTH)
    render.set_image(640, 480, intrinsic(), extrinsic(info['camLoc']))

    # ------------------------------------------------------------------
    # Pass 2: process only valid frames (mask not touching any border)
    # ------------------------------------------------------------------
    seq_start = time.time()
    n_kept    = 0

    for frame_idx, at_edge in enumerate(at_edge_flags):
        ok_rgb, rgb = rgb_cap.read()
        ok_seg, seg = seg_cap.read()

        if not ok_rgb or not ok_seg:
            break

        if at_edge:
            continue

        # Mask for crop parameters
        mask = seg[:, :, 0] > 0
        cx, cy, half = get_square_crop(mask)

        # Render depth map
        V, F = reader.read_human(sample, frame_idx)
        F = np.array(F)

        for garment in garments:
            _V = reader.read_garment_vertices(sample, garment, frame_idx)
            _F = reader.read_garment_topology(sample, garment)
            _F = quads2tris(_F)
            F  = np.concatenate((F, _F + V.shape[0]), axis=0)
            V  = np.concatenate((V, _V), axis=0)

        render.set_mesh(V, F)
        raw   = render.render()
        depth = np.array(raw).squeeze()
        depth[depth >= MAX_DEPTH - 1] = 0.0

        # Crop
        rgb_crop   = crop_image(rgb,   cx, cy, half)
        depth_crop = crop_image(depth, cx, cy, half)

        # Save
        name = f'{sample}_{frame_idx}'
        cv2.imwrite(os.path.join(IMAGE_DIR, name + '.jpg'), rgb_crop)
        np.save(os.path.join(DEPTH_DIR, name + '.npy'), depth_crop)
        saved.append(name)
        n_kept += 1

    rgb_cap.release()
    seg_cap.release()

    # ------------------------------------------------------------------
    # Progress print with timing and ETA
    # ------------------------------------------------------------------
    seq_elapsed   = time.time() - seq_start
    total_elapsed = time.time() - global_start
    seqs_done     = seq_idx + 1
    seqs_left     = total_seqs - seqs_done
    avg_per_seq   = total_elapsed / seqs_done
    eta           = avg_per_seq * seqs_left

    print(
        f'  [{seqs_done:3d}/{total_seqs}] {sample} | '
        f'kept {n_kept}/{n_total} frames | '
        f'seq {fmt_duration(seq_elapsed)} | '
        f'elapsed {fmt_duration(total_elapsed)} | '
        f'ETA {fmt_duration(eta)}'
    )
    return saved


# ---------------------------------------------------------------------------
# Split folders
# ---------------------------------------------------------------------------
all_folders = sorted(
    f for f in os.listdir(SUBSET_DIR)
    if os.path.isdir(os.path.join(SUBSET_DIR, f))
)

train_folders = all_folders[:128]    # 00001-00152
val_folders   = all_folders[128:144] # 00153-00168
test_folders  = all_folders[144:]    # 00169-00185


def write_txt(path, names):
    with open(path, 'w') as f:
        f.write('\n'.join(names))
        if names:
            f.write('\n')


def run_split(label, folders, global_start, offset=0):
    names = []
    total = len(folders)
    print(f'\nProcessing {total} {label} sequences...')
    for i, folder in enumerate(folders):
        names.extend(process_sequence(folder, offset + i,
                                      len(all_folders), global_start))
    return names


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    global_start = time.time()

    train_names = run_split('training',   train_folders, global_start,
                            offset=0)
    val_names   = run_split('validation', val_folders,   global_start,
                            offset=len(train_folders))
    test_names  = run_split('test',       test_folders,  global_start,
                            offset=len(train_folders) + len(val_folders))

    write_txt(os.path.join(OUT_DIR, 'train.txt'),      train_names)
    write_txt(os.path.join(OUT_DIR, 'validation.txt'), val_names)
    write_txt(os.path.join(OUT_DIR, 'test.txt'),       test_names)

    total = time.time() - global_start
    print(f'\nDone in {fmt_duration(total)}.')
    print(f'  Train      : {len(train_names)} frames')
    print(f'  Validation : {len(val_names)} frames')
    print(f'  Test       : {len(test_names)} frames')
