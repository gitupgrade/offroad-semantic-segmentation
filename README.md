NOTE:THIS PROGRAM HAS BEEN COMPLETELY WRITTEN ON LINUX UBUNTU + NVIDIA GPU(CUDA WORKING)
Cpu-only training may not be supported.
# Offroad Semantic Segmentation – Duality AI Hackathon

This repository contains our final submission for the **Offroad Semantic Segmentation** track of the Duality AI Hackathon.  
The task is to segment off-road desert scenes using synthetic data generated from Duality AI’s Falcon digital twin platform.



---

## Task
- **Problem:** Semantic scene segmentation for off-road autonomy
- **Dataset:** Organizer-provided synthetic desert dataset
- **Splits:**
  - `train/` – used for training only
  - `val/` – used for evaluation only (unseen during training)

---

## Model
- **Architecture:** DeepLabV3
- **Backbone:** MobileNetV3
- **Number of classes:** 10
- **Loss:** Cross-Entropy Loss (with class weights)
- **Framework:** PyTorch

---

## Evaluation Metric
- **Metric:** Mean Intersection over Union (mIoU)
- **Evaluation set:** Organizer-provided validation set (`val/`)
- **Final validation mIoU:** **0.7646**


---

## Repository Structure
```
.
├── config/ # Configuration files
├── models/ # Model definitions
├── utils/ # Dataset loader, training utilities, metrics
├── runs/ # Trained model weights (.pt)
├── train.py # Training script
├── test.py # Evaluation script (validation set)
├── requirements.txt # Dependencies
└── README.md
```
---
## Dataset Folder Structure (Required)

Your dataset path (`DATA_PATH` in `config/config.yaml`) **must** have this exact structure:
```
DATA_PATH/
├── train/
│   ├── Color_Images/       # RGB input images
│   └── Segmentation/       # segmentation masks (same filename as image)
└── val/
    ├── Color_Images/
    └── Segmentation/
```
**Important:** Each image in `Color_Images/` must have a corresponding mask in `Segmentation/` with the **same filename** (only extension may differ).
Example: `xxx.png` ↔ `xxx.png`
---


## Setup
```
git clone https://github.com/gitupgrade/offroad-semantic-segmentation.git
cd offroad-semantic-segmentation
ls   # should show train.py, test.py, requirements.txt
```
1. Create a Python environment
  

---




AT EACH STEP PLEASE MAKE SURE YOU ARE IN RIGHT DIRECTORY IN TERMINAL.
---

#To set up:
```
conda create -n nm_seg python=3.10 -y
conda activate nm_seg

```
---
2. Install dependencies:
```
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
```
---
#Update the dataset path in:
```
config/config.yaml
```
Set DATA_PATH to the local dataset directory.

---
```
export WANDB_MODE=offline
```
---
#Training
 Train using:
```

python train.py
```
---
#Run evaluation using
```
python test.py
```
---

#Notes
1.The validation set was not used for training
2.No test data was used or accessed by the team
3.Model weights are provided in the runs/ directory
4.Absolute paths are not stored in checkpoints; paths are configurable via config.yaml




















