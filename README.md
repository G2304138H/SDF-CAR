# SDF-CAR: 3D Coronary Artery Reconstruction from Two Views with a Hybrid SDF-Occupancy Implicit Representation

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

**SDF-CAR** is a novel self-supervised framework for reconstructing 3D coronary arteries from only two sparse 2D X-ray projections. By leveraging a **Hybrid SDF-Occupancy** representation, we overcome the "blobby" artifacts and broken connectivity common in pure occupancy networks (like NeCA), achieving state-of-the-art topological accuracy.

<p align="center">
  <img src="assets/3D_SDF_to_2D.png" alt="SDF-CAR Pipeline" width="100%">
  <br>
  <em>Figure 1: Overview of the SDF-CAR Framework.</em>
</p>

## 📄 Abstract

**Note: This paper is currently under review.**

The three-dimensional (3D) reconstruction of coronary arteries is crucial for diagnosis but difficult to achieve from standard Invasive Coronary Angiography (ICA) which provides only sparse 2D views. We propose **SDF-CAR**, a self-supervised framework that leverages a **Signed Distance Field (SDF)**-based neural implicit representation. Unlike supervised methods that require unavailable 3D ground truth, SDF-CAR optimizes a patient-specific model directly from 2D projections. By integrating SDF-based geometric priors with an occupancy-based differentiable rendering loss, we improve the **Centerline Dice (cIDice)** score by over **16%** compared to state-of-the-art baselines, ensuring smooth, continuous vessel reconstruction.

## 🏆 Key Features

* **Hybrid Representation:** Combines the optimization stability of Occupancy networks with the geometric surface precision of Signed Distance Functions (SDF).
* **Sparse View Reconstruction:** Works effectively with only **2 standard angiographic views**.
* **Topological Preservation:** Significantly reduces broken vessel segments and disconnected branches in distal areas.
* **Self-Supervised:** No 3D ground truth required for training; optimizes directly on patient projection data.

## 📊 Qualitative Results

### Right Coronary Artery (RCA)
SDF-CAR maintains connectivity in complex curved segments where baselines often fail.

<p align="center">
  <img src="assets/RCA_Diagrams.png" alt="RCA Qualitative Results" width="100%">
</p>

### Left Anterior Descending (LAD)
Our method successfully captures fine distal branches that are often missed by occupancy-only methods.

<p align="center">
  <img src="assets/LAD_Diagrams.png" alt="LAD Qualitative Results" width="100%">
</p>

## 🛠️ Installation

This code is based on PyTorch and requires Linux with an NVIDIA GPU, a recent
CUDA-capable driver, the CUDA toolkit/`nvcc` for the bundled hash-grid extension,
and Python 3.10 or newer. Conda is not required.

```bash
# 1. Create and activate a standard Python virtual environment
cd SDF-CAR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 2. Install a CUDA-enabled PyTorch wheel
# Choose the wheel supported by your NVIDIA driver from https://pytorch.org/get-started/locally/.
# Example for a driver supporting CUDA 12.6:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install this repository's remaining Python dependencies
python -m pip install -r requirements.txt

# 4. Verify the GPU packages before training
python -c "import torch, astra; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); astra.test()"
nvcc --version
```

The host's installed CUDA toolkit need not exactly equal PyTorch's wheel label;
the NVIDIA driver must support the wheel's runtime. If `nvidia-smi` reports only
CUDA 12.5 support, select a CUDA 12.4 PyTorch build from the
[official previous-versions page](https://pytorch.org/get-started/previous-versions/)
instead of the `cu126` example.

## 📂 Data Preparation

We utilize the **ImageCAS** dataset (Coronary Artery Segmentation from CCTA).

1.  Download the ImageCAS dataset.
2.  Preprocess the CCTA volumes to extract coronary artery segmentations.
3.  Organize data as follows:

    ```
    data/
    ├── LAD_GT/          # Ground truth LAD segmentations
    │   ├── 900.npy
    │   ├── 901.npy
    │   └── ...
    ├── RCA_GT/          # Ground truth RCA segmentations
    │   ├── 900.npy
    │   ├── 901.npy
    │   └── ...
    └── ...
    ```

4.  Update the configuration file `config/CCTA.yaml` with the correct data paths.

## 🚀 Usage

### Single Model Training
To train on a single model:

```bash
python train.py --config config/CCTA.yaml
```

### Direct per-scene optimization from segmented-view NPZ

The original `CCTA.yaml` path loads a 3D volume and synthesizes its own
projections. For the intended reconstruction setting, configure an external
projection NPZ instead. The NPZ must contain `images [V,H,W]`, `theta_deg [V]`,
`phi_deg [V]`, `sid`, and `imager_pixel_spacing`; the Stage-2 camera convention
uses a 0.75 m source-to-isocentre distance.

The supplied case is configured in `config/CCTA_npz_case1.yaml`. It supports a
two-stage, same-YAML geometry-validation workflow. First generate training
views from the reference voxel mask with the exact ODL/ASTRA cameras, then run
optimization from the generated NPZ:

```bash
python generate_2d_projections.py --config config/CCTA_npz_case1.yaml
python train.py --config config/CCTA_npz_case1.yaml
```

`projection_generation.camera_metadata_npz` supplies the camera angles, SID,
detector spacing, clinical labels, and projection centre. The generator ignores
that file's existing images. It samples `reference_volume_npz["vol"]` into the
configured centred reconstruction grid, forward-projects it with
`ODL ConeBeamGeometry`/`astra_cuda`, and writes `exp.projection_npz` plus PNGs
for the two selected views. The NPZ uses the raw ray lengths as its training
`images`, matching the original SDF-CAR synthetic-projection path; it also keeps
binary silhouettes under `binary_images` for inspection. The original
camera-metadata NPZ is not overwritten.
Training reads the generated `exp.projection_npz` from the same YAML, selects
views 0 and 1, performs a fresh patient-specific optimization, and does not load
a pretrained checkpoint.

This generated-view mode is a controlled geometry/optimization validation: it
uses the reference 3D mask to create the 2D targets, so it must not be reported
as real two-view inference. For real inference, point `exp.projection_npz`
directly at the acquired/segmented view NPZ and do not run the generator.

For the supplied `2d_1.npz`, a direct two-view configuration is provided:

```bash
python train.py --config config/CCTA_npz_direct_case1.yaml
```

This selects views 0 and 1 from the input NPZ and uses their masks directly;
`voxel_1.npz` is consulted only after optimization for full-grid output and
DSC. Change the two absolute NPZ paths after cloning on another machine.

Camera conversion is performed in the centred XYZ reference frame used by
ODL. A Stage-2 pair `(theta, phi)` has central ray
`[sin(phi)cos(theta), sin(phi)sin(theta), cos(phi)]`. Its direction-equivalent
SDF-CAR YAML pair is `[phi - 90, 90 - theta]` degrees. The selected supplied
views therefore map as follows:

```text
theta=-40, phi=80 -> SDF-CAR [-10, 130]
theta= 75, phi=80 -> SDF-CAR [-10,  15]
```

Training passes the complete camera frame to `ODL ConeBeamGeometry`: the
source-to-detector direction plus detector row and column axes. The two-angle
values are saved for audit, but are not used alone because they cannot preserve
detector roll. The reconstruction NPZ records these as
`sdfcar_projection_angles_deg`, `odl_source_to_detector_unit_xyz`,
`odl_detector_row_axis_xyz`, and `odl_detector_column_axis_xyz`.

The one-view ODL motion partition is centred on the requested angle. This fixes
an upstream issue where `uniform_partition(0, requested_angle, shape=1)` samples
the midpoint (`requested_angle / 2`) rather than the requested view.

### Trainer stability and diagnostics

SDF mode initializes the final network bias to `0.1` by default. With
`sdf_alpha: 50`, this starts at approximately `0.0067` occupancy rather than a
half-filled volume whose ray integrals saturate the binary silhouette loss.
Configure this with `network.sdf_initial_bias`.

Only the neural network uses mixed precision. ODL/ASTRA projections, the
ray-integral-to-mask exponential, the differentiable distance transform, and
both losses run in FP32. A nonzero 2D SDF loss now requires `kornia`; the trainer
fails with an installation command instead of silently using the detached
SciPy fallback.

Every run creates `training_log_<current_model_id>.jsonl`. Its first record
explicitly lists the trainer fixes active for that run. Each epoch then records
the two loss components, occupancy and projection ranges, gradient norm,
parameter-probe change, AMP scale, and whether AMP skipped the optimizer step.
The console prints the same critical values. If the gradient is dead for
`train.zero_gradient_patience` consecutive epochs, training aborts and points
to this log instead of completing a meaningless constant-loss run.

Kornia's soft distance transform contains a logarithm, so the trainer bounds
its input with `train.sdf_distance_epsilon` (default `1e-6`) to keep gradients
finite at exact-zero detector margins. The geometric SDF loss is disabled for
`train.sdf_loss_warmup_epochs` (default 100) while the projection term forms a
silhouette, then reaches its configured weight over
`train.sdf_loss_ramp_epochs` (default 400). The effective weight is printed and
written to the JSONL log each epoch. A non-finite epoch never updates AdamW.

During output evaluation, `reference_volume_npz` maps the centred reconstruction
ROI back to the reference XYZ grid and supplies the GT mask for DSC.

The combined output is saved as
`reconstruction_<current_model_id>.npz`. It contains the raw XYZ SDF ROI, the
soft occupancy ROI, the thresholded ROI mask, camera/input metadata, and `vol`.
When a reference NPZ with `vol` and `spacing` is configured, output `vol` has
that reference shape and is a binary `uint8` XYZ voxel array.

Per-case timing is recorded with CUDA synchronization at each measurement
boundary. `optimization_time_seconds`, `mean_epoch_time_seconds`, initialization,
evaluation, and total compute time are embedded in the reconstruction NPZ. A
human-readable `timing_<current_model_id>.json` is saved beside it and also
reports end-to-end wall time, GPU model, and peak allocated GPU memory.

When `reference_volume_npz` is supplied, the final full-grid prediction is
compared with `reference_volume_npz["vol"] > 0`. The voxel mask Dice score
`2 * intersection / (prediction + reference)` is printed and saved as
`mask_dsc` in both the reconstruction NPZ and timing JSON. Foreground and
intersection voxel counts are saved alongside it for verification.

Visual quality-control artifacts are also generated after direct optimization:
`predicted_projection_view_<index>.png` for each selected input camera, plus
`3d_ground_truth_surface.gif` and `3d_prediction_surface.gif` when a reference
volume is available. The two surface GIFs match the parametric evaluator's
single-surface camera path (24 frames, 5 FPS, one 360-degree azimuth rotation,
with sinusoidal elevation from 16 to 28 degrees). These settings can be changed
under `visualization` in the case YAML.

Direct optimization still requires an NVIDIA CUDA system: the hash-grid
encoder compiles a CUDA extension and the forward projector uses
`astra_cuda`. It cannot run on CPU-only or Apple Silicon environments.

### Batch Training
To train on multiple models automatically:

```bash
python batch_train.py --config config/CCTA.yaml --num_gpus 1
```

The training script will automatically process models specified in the `model_numbers` list in the config file.

### Evaluation
To evaluate reconstructions against ground truth:

```bash
python eval.py --pred_dir logs/reconstructions/ --gt_dir data/LAD_GT/ --output_dir results/
```

### Visualization
To visualize and compare reconstructions:

```bash
python vis.py
```

## 📝 Citation

**Paper Under Review**

Citation information will be updated upon publication acceptance.

## 🙏 Acknowledgements

This code heavily builds upon the following excellent repositories:

* [NeCA](https://github.com/SID-CoroRecon/NeCA)
* [Instant-NGP](https://github.com/NVlabs/instant-ngp)
* [ImageCAS Dataset](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT)

---
For questions, please contact:

[es-AhmedR.Ali2025@alexu.edu.eg]

[es-MohamedA.Hamdy2025@alexu.edu.eg].
