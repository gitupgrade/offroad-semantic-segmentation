NOTE:THIS PROGRAM HAS BEEN COMPLETELY WRITTEN ON LINUX UBUNTU.
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

## Setup
1. Create a Python environment
   > Note: The provided `requirements.txt` reflects the development environment; minimal dependencies are sufficient to run training and evaluation .Its  up to you to download all mentioned in requirements or just minimal required to run the model.
2. Install dependencies:
```
pip install -r requirements.txt
```
---
#Update the dataset path in:
```
config/config.yaml
```

Set DATA_PATH to the local dataset directory.

AT EACH STEP PLEASE MAKE SURE YOU ARE IN RIGHT DIRECTORY IN TERMINAL.
---

#To set up:
```
conda activate nm_seg
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














