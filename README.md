# CLOTH3D++ Body and Cloth Depth Estimation

This repository contains the code and experiments for Deliverable 3 of the OR assignment: estimating body-and-cloth depth from RGB images from the CLOTH3D++ dataset.

The project covers the complete workflow:

- rendering depth maps from CLOTH3D++ body and garment meshes;
- removing frames whose subject touches the image border;
- cropping RGB and depth around the subject;
- comparing a UNet baseline with an ImageNet-pretrained DeiT-Small model;
- evaluating weak and geometry-aware losses;
- projecting 14 SMPL joints into the image and using them for multi-task supervision;
- fine-tuning a final model and comparing it with earlier stages.

## Repository layout

```text
preprocess.py                 Render, crop, split, and save RGB/depth pairs
resizing.py                   Standalone 256x256 resizer
demo_cloth3d.ipynb            CLOTH3D starter-kit/DataReader demo
depth_estimation.ipynb        Original TensorFlow UNet baseline

depth_estimation_d1.ipynb     Task 1: CNN/UNet experiments
depth_estimation_d2.ipynb     Task 2: DeiT transformer experiments
depth_estimation_d3.ipynb     Task 3: weak and geometry-aware losses
depth_estimation_d4.ipynb     Task 4: final integration

assignment.pdf                 Assignment specification
OR Deliverable 1 Report.pdf    Earlier assignment report
OR Deliverable 2 Report.pdf    Earlier assignment report

cloth3d/                       CLOTH3D starter kit and DataReader
cloth3d++_subset/              CLOTH3D++ subset, one folder per sequence
preprocessed_data/             Cropped image/depth pairs and split files
runs/                          Metrics, figures, decision files, and run artifacts
```

Generated checkpoints, histories, joint caches, and resized data are excluded by `.gitignore` because they can be large.

## Requirements

Use Python 3.10 or newer and preferably a CUDA-enabled PyTorch installation for training. Install the base dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The experiment notebooks additionally use packages that are not currently listed in `requirements.txt`:

```powershell
python -m pip install torch torchvision timm torchinfo imageio pandas matplotlib
```

`preprocess.py` uses the `deodr` renderer through the bundled `cloth3d/DataReader` code. The notebooks use PyTorch and `timm`; a GPU is strongly recommended for D1-D4.

## Data preparation

Place the CLOTH3D++ subset in the repository root so that these files exist:

```text
cloth3d++_subset/00001/00001.mkv
cloth3d++_subset/00001/00001_segm.mkv
cloth3d++_subset/00001/info.mat
...
```

Run preprocessing from the repository root:

```powershell
python preprocess.py
```

For every sequence, `preprocess.py` scans the segmentation video, discards frames whose mask touches an image border, renders body and garment depth, applies a square crop with a 10-pixel margin, and writes:

```text
preprocessed_data/
  image/<sample>_<frame>.jpg
  depth/<sample>_<frame>.npy
  train.txt
  validation.txt
  test.txt
```

The split follows the assignment:

- training: folders `00001` to `00152`;
- validation: folders `00153` to `00168`;
- test: folders `00169` to `00185`.

The recorded preprocessing run contains 30,325 training frames, 3,283 validation frames, and 2,827 test frames.

## Resizing

The standalone script can create nearest-neighbour and bilinear 256x256 copies:

```powershell
python resizing.py
```

It writes to `resized_data/`. The D1-D4 notebooks use their experiment-specific variant layout under `preprocessed_data/resized_variants/`, typically including `256_nearest`, `256_bilinear`, and `384_nearest`. Run the notebooks' preprocessing cells when those variants are not already present.

For depth, nearest-neighbour interpolation preserves the zero-valued background mask. Bilinear interpolation is retained as an experiment because it can produce different boundary behaviour.

## Experiments

Run the notebooks in order from the repository root. In VS Code, select the same Python environment for each notebook.

### D1: UNet baseline

`depth_estimation_d1.ipynb` evaluates:

1. nearest-neighbour versus bilinear resizing;
2. augmentation ablations;
3. learning rate and batch size;
4. 256 versus 384 resolution.

The selected CNN pipeline is 256x256, nearest-neighbour resizing, no augmentation, learning rate `1e-3`, and batch size `8`. Results are written to `runs/d1_unet/`.

### D2: DeiT transformer

`depth_estimation_d2.ipynb` uses a pretrained DeiT-Small encoder and compares:

1. bilinear and FPN decoders;
2. patch sizes 16 and 8;
3. full fine-tuning and a frozen backbone.

The best recorded D2 configuration is DeiT-Small with patch size 8, a bilinear decoder, full fine-tuning, learning rate `1e-4`, and batch size `4`. Results are written to `runs/d2_deit/`.

### D3: weak and geometry-aware losses

`depth_estimation_d3.ipynb` caches projected 2D coordinates for 14 SMPL joints and compares:

- supervised depth L1;
- weak-only ranking and normal consistency losses;
- depth plus scale-invariant and joint losses;
- a combined loss regime.

Results and cached joints are written to `runs/d3_losses/`. The raw sequence metadata and videos in `cloth3d++_subset/` are required for this stage.

### D4: final integration

`depth_estimation_d4.ipynb` starts from the best D3 checkpoint and performs focused fine-tuning experiments with lower learning rate and geometry-aware loss weights. Results are written to `runs/d4_final/`.

The recorded best single model is `d4_l1_geom_small_deit_p8_bs4_lr3e_5`.

## Recorded results

The metrics below are from the checked-in CSV result summaries. RMSE and normal MAE are reported in the notebooks' normalized depth space.

| Stage | Best configuration | Test RMSE | Normal MAE | Joint L1 (px) | FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| D1 UNet | 256 nearest, no augmentation, lr 1e-3, batch 8 | 0.0660 | 1.1281 | - | 179.7 |
| D2 DeiT | patch 8, bilinear decoder, full fine-tuning | 0.0537 | 1.0153 | - | 61.5 |
| D3 losses | DeiT patch 8, supervised depth L1 | 0.0494 | 1.2798 | 162.7 | 61.2 |
| D4 final | DeiT patch 8, depth L1 + small geometry weights | 0.0497 | 1.2786 | 13.0 | 61.6 |

D4 substantially improves the projected-joint metric while maintaining depth RMSE close to the D3 result. Use the JSON decision files in each `runs/` subdirectory for the exact selected checkpoints and loss weights.

## Evaluation and outputs

Each experiment stores some combination of:

- `*_results.csv`: tabular run summaries;
- `*_stage_decisions.json`: selected configurations;
- `metrics/`: per-run JSON metrics;
- `figures/`: qualitative predictions, error maps, and training plots;
- `checkpoints/`: model weights, ignored by Git;
- `histories/`: training histories, ignored by Git.

The primary metrics are depth RMSE, surface-normal MAE, and, where applicable, 2D joint L1 error. The reported values depend on the notebooks' depth normalization and should be compared only within the same preprocessing and evaluation convention.

## Notes

- Run notebooks with the repository root as the current working directory. They resolve data and output paths from `Path.cwd()`.
- Windows notebook execution uses `NUM_WORKERS = 0` to avoid multiprocessing issues.
- `demo_cloth3d.ipynb` is a starter-kit demonstration and is not part of the D1-D4 training pipeline.
- The original `depth_estimation.ipynb` is the older TensorFlow/keras-unet baseline; D1-D4 are the local PyTorch experiment notebooks.
- The dataset, SMPL model files, rendered data, and checkpoints are not reproduced by this README and may be subject to their original dataset or course terms.
