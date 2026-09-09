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

This code is based on PyTorch and requires a GPU with CUDA support.

```bash
# 1. Navigate to the project directory
cd SDF-CAR

# 2. Create a conda environment
conda create -n sdf-car python=3.8
conda activate sdf-car

# 3. Install PyTorch (Adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install required dependencies
pip install numpy scipy pyyaml tqdm matplotlib pandas
pip install odl tigre

# 5. (Optional) Install tiny-cuda-nn for hash encoding acceleration
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

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

The supplied case is configured in `config/CCTA_npz_case1.yaml`:

```bash
python train.py --config config/CCTA_npz_case1.yaml
```

This selects views 0 and 1, performs a fresh patient-specific optimization,
and does not load a pretrained checkpoint. `reference_volume_npz` is optional
and is used only after optimization to map the centred reconstruction ROI back
to the reference XYZ grid. It is never used to form the optimization targets.

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
