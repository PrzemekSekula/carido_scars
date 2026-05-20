# YOLO Scar Detection

This folder contains scripts to employ YOLO for detecting scars on images.

## Structure
- `alg_yolo.py`: Contains the `YOLOAlgorithm` class which encapsulates training and inference logic using the `ultralytics` library.
- `exp_yolo.py`: The main execution script to run training or inference.

## Prerequisites
Install the required library:
```bash
pip install ultralytics
```

## Dataset Preparation
YOLO requires data in a specific format. You need to create a `data.yaml` file and organize your images/labels as follows:

### Directory Structure
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

### `data.yaml` Example
```yaml
path: ../dataset  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
names:
  0: scar
```

## Usage

### 1. Running Inference (Pre-trained Model)
To run detection on your images using a pre-trained model (defaults to `data/surgwound/IMAGES`):
```bash
python YOLO/exp_yolo.py --mode predict
```

If you have a custom pre-trained model file:
```bash
python YOLO/exp_yolo.py --mode predict --model path/to/your_model.pt
```

### 2. Training (Requires custom labels)
To train the model on your custom dataset once you have annotated your data:
```bash
python YOLO/exp_yolo.py --mode train --data path/to/data.yaml --epochs 50 --batch 16
```

The results will be saved in the `YOLO/runs` directory.
