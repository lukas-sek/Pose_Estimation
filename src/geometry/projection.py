import numpy as np


def project_points(camera_matrix, points_3d):
    ones = np.ones((points_3d.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate((points_3d, ones), axis=1)
    projected = (camera_matrix @ homogeneous.T).T
    depth = projected[:, 2]
    return projected[:, :2] / (depth[:, None] + 1e-9), depth


def crop_bounds_from_mask(mask, margin=10):
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    columns = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(columns)[0][[0, -1]]
    center_x = int((col_min + col_max) / 2)
    center_y = int((row_min + row_max) / 2)
    half = max(row_max - row_min, col_max - col_min) // 2 + margin
    height, width = mask.shape[:2]
    return max(0, center_x - half), max(0, center_y - half), min(width, center_x + half), min(height, center_y + half)
