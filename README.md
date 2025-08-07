# DualPhys-GS: Dual Physically-Guided 3D Gaussian Splatting for Underwater Scene Reconstruction

<div align="left">
Jiachen Li, Guangzhi Han, Jin Wan, Yuan Gao, Delong Han

</div>

---

## Abstract

In 3D reconstruction of underwater scenes, traditional methods based on atmospheric optical models cannot effectively deal with the selective attenuation of light wavelengths and the effect of suspended particle scattering, which are unique to the water medium, and lead to color distortion, geometric artifacts, and collapsing phenomena at long distances. We propose the DualPhys-GS framework to achieve high-quality underwater reconstruction through a dual-path optimization mechanism. Our approach further develops a dual feature-guided attenuation-scattering modeling mechanism, the RGB-guided attenuation optimization model combines RGB features and depth information and can handle edge and structural details. In contrast, the multi-scale depth-aware scattering model captures scattering effects at different scales using a feature pyramid network and an attention mechanism. Meanwhile, we design several special loss functions. The attenuation scattering consistency loss ensures physical consistency. The water body type adaptive loss dynamically adjusts the weighting coefficients. The edge-aware scattering loss is used to maintain the sharpness of structural edges. The multi-scale feature loss helps to capture global and local structural information. In addition, we design a scene adaptive mechanism that can automatically identify the water-body-type characteristics (e.g., clear coral reef waters or turbid coastal waters) and dynamically adjust the scattering and attenuation parameters and optimization strategies. Experimental results show that our method outperforms existing methods in several metrics, especially in suspended matter-dense regions and long-distance scenes, and the reconstruction quality is significantly improved.

---

## Pipeline

![pipeline](https://github.com/Hangz0416/Dualphys-GS/blob/master/assets/pipeline.png)" style="zoom:100%;" />

## Installation

### Prerequisites

- Ubuntu 22.04
- CUDA 11.8
- Python 3.10

### Quick Setup

```bash
# Clone repository with submodules
git clone git@github.com:HarveyMu/Dualphys-GS.git --recursive
cd Dualphys-GS

# Create conda environment
conda create --name dualphys_gs python=3.10 -y
conda activate dualphys_gs

# Install PyTorch with CUDA support
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Build custom CUDA modules
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

---

## Dataset Preparation

### Available Datasets

| Dataset                | Download                                                                                        |
| :--------------------- | ----------------------------------------------------------------------------------------------- |
| **SaltPond**     | [Google Drive](https://drive.google.com/file/d/1gItZkfEFmXZzIRh5b6wXeWD6GappX-QN/view?usp=sharing) |
| **SeaThru-NeRF** | [Google Drive](https://drive.google.com/uc?export=download&id=1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT)   |

### Data Structure

Expected directory structure for training:

```
your_dataset/
├── images/           # Undistorted RGB images
├── sparse/
│   └── 0/           # COLMAP reconstruction
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── (optional) masks/ # Object masks for advanced training
```

### SeaThru-NeRF Preprocessing

SeaThru-NeRF uses `OPENCV` camera models, but our rasterizer requires `SIMPLE_PINHOLE` or `PINHOLE`. Convert as follows:

1. **Organize files**:

   ```
   <scene_path>/
   ├── input/           # Original images
   │   ├── image_001.jpg
   │   └── ...
   └── distorted/       # Original COLMAP data
       ├── database.db
       └── sparse/0/
   ```
2. **Convert camera models**:

   ```bash
   python convert.py -s <scene_path> --skip_matching [--resize]
   ```

---

## Quick Start



### Basic Training

```bash
python train.py \
    -s /path/to/your/dataset \
    --exp experiment_name \
    --do_seathru \
    --seathru_from_iter 10000
```

### Advanced Configuration

For fine-tuned control, modify parameters in `arguments/__init__.py` or use command-line arguments:

```bash
python train.py \
    -s /path/to/dataset \
    --exp advanced_experiment \
    --do_seathru \
    --seathru_from_iter 10000 \
    --use_rgb_guided_at \
    --use_multiscale_bs \
    --use_edge_aware_bs_loss \
    --iterations 30000
```
## Acknowledgments
The codebase is built upon the original Yang Seasplat [implementation](https://github.com/dxyang/seasplat/). We sincerely thank the authors of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), [Seathru-NeRF](https://sea-thru-nerf.github.io/), [Seasplat](https://github.com/dxyang/seasplat/), whose codes and datasets were used in our work.
