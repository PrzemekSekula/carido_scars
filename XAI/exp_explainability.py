"""
Script to run Grad-CAM explainability on the Trained Model
This script loads the trained model and dataset, picks random samples, and generates Grad-CAM heatmaps.

Autor: Przemek Sekula
Created: 2026-01-20
"""

import sys
import os
from pathlib import Path

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ImageClassifier
from data_tools import ImageDataModule
from util_explainability import GradCAM, overlay_heatmap
import cv2

# Hardcoded config for convenience (should match training config generally)
class Config:
    data_dir = r"..\\Data\\Data" # Adjusted based on actual file structure
    # Note: we will try to use path relative to the script location if possible or assume standard layout
    # For now, let's trust the DataModule can be instantiated.
    # Actually, we can get params from the checkpoint.
    batch_size = 1
    num_workers = 0
    test_split_size = 0.2
    val_split_size = 0.2
    image_size = [380, 380] # EfficientNet B4 size
    use_class_weights = False

def run_explainability():
    # 1. Setup Paths
    ckpt_path = r"d:\iitis\medical\carido_scars\checkpoints\All\tf_efficientnet_b4\tf_efficientnet_b4-best-checkpoint.ckpt"
    # Output path relative to this script's directory
    output_dir = Path(__file__).parent / "explanation_gradCAM"
    output_dir.mkdir(exist_ok=True)
    
    # 2. Load Model
    print(f"Loading model from {ckpt_path}...")
    try:
        model = ImageClassifier.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    # 3. Identify Target Layer
    # For tf_efficientnet_b4 in timm, the last convolutional layer before the classifier is usually 'conv_head' 
    # or part of the last block. 'conv_head' is a good candidate.
    target_layer = model.backbone.conv_head
    print(f"Target layer for Grad-CAM: {target_layer}")
    
    # 4. Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 5. Load Data
    # We need to recreate the DataModule to get the test set. 
    # We can use the params from the loaded model if available, or fall back to defaults.
    config = Config()
    # Attempt to use data_dir from hparams if it exists, else default
    # if hasattr(model.hparams, 'data_dir'):
    #     config.data_dir = model.hparams.data_dir
    if hasattr(model.hparams, 'image_size'):
        config.image_size = model.hparams.image_size
        
    print(f"Loading data from {config.data_dir}...")
    dm = ImageDataModule(config)
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    
    # 6. Generate Explanations
    print("Generating explanations for all test samples...")
    
    # Get all test samples
    test_samples = list(test_loader)
    
    # We will iterate through a few batches (batch_size is 1 here effectively due to how we loop, 
    # but let's just take a few images from the first batch if batch_size > 1)
    
    count = 0
    # max_count = 10 # Process all

    
    for batch_idx, (images, labels) in enumerate(test_loader):
        for i in range(images.size(0)):
            # if count >= max_count:
            #     break
                
            img_tensor = images[i:i+1] # Keep batch dim: (1, C, H, W)
            real_label = labels[i].item()
            
            # Predict
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                predicted_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_label].item()
            
            # Generate Heatmap
            # We explain the PREDICTED class
            heatmap, explained_class = grad_cam(img_tensor, class_idx=predicted_label)
            
            # Overlay
            overlay, original = overlay_heatmap(img_tensor, heatmap)
            
            # Save
            filename = f"sample_{count:02d}_real_{real_label}_pred_{predicted_label}_conf_{confidence:.2f}.jpg"
            save_path = output_dir / filename
            
            # Create a composite image with matplotlib
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            ax[0].imshow(original)
            ax[0].set_title(f"Original (Real: {real_label})")
            ax[0].axis('off')
            
            ax[1].imshow(heatmap, cmap='jet')
            ax[1].set_title("Grad-CAM Heatmap")
            ax[1].axis('off')
            
            ax[2].imshow(overlay)
            ax[2].set_title(f"Overlay (Pred: {predicted_label}, Conf: {confidence:.2f})")
            ax[2].axis('off')
            
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved explanation to {save_path}")
            
            count += 1
        
        # if count >= max_count:
        #     break

    print("\nDone! Please check the 'explanations' folder.")

if __name__ == "__main__":
    run_explainability()
