"""
Script to run Occlusion Sensitivity explainability on the Trained Model.
Generates perturbation-based attribution maps.

Autor: Przemek Sekula (Antigravity)
Created: 2026-01-20
"""

import sys
import os
from pathlib import Path

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'initial_experiments'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ImageClassifier
from data_tools import ImageDataModule
from util_explainability import OcclusionSensitivity, overlay_heatmap
import cv2

# Hardcoded config for convenience behavior
class Config:
    data_dir = r"..\\Data\\Data" 
    batch_size = 1
    num_workers = 0
    test_split_size = 0.2
    val_split_size = 0.2
    image_size = [380, 380] # EfficientNet B4 size
    use_class_weights = False

def run_occlusion():
    # 1. Setup Paths
    ckpt_path = r"d:\iitis\medical\carido_scars\checkpoints\All\tf_efficientnet_b4\tf_efficientnet_b4-best-checkpoint.ckpt"
    # Output path relative to this script's directory
    output_dir = Path(__file__).parent / "explanation_occlusion"
    output_dir.mkdir(exist_ok=True)
    
    # 2. Load Model
    print(f"Loading model from {ckpt_path}...")
    try:
        model = ImageClassifier.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    # 3. Initialize Occlusion Sensitivity
    occlusion = OcclusionSensitivity(model)
    
    # 4. Load Data
    config = Config()
    if hasattr(model.hparams, 'image_size'):
        config.image_size = model.hparams.image_size
        
    print(f"Loading data from {config.data_dir}...")
    dm = ImageDataModule(config)
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    
    # 5. Generate Explanations
    print("Generating Occlusion Sensitivity maps for all test samples...")
    print("Note: This method is slower than Grad-CAM.")
    
    count = 0
    # max_count = 5 # Uncomment for testing
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        for i in range(images.size(0)):
            # if count >= max_count: break
                
            img_tensor = images[i:i+1] # Keep batch dim: (1, C, H, W)
            real_label = labels[i].item()
            
            # Predict
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                predicted_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_label].item()
            
            # Generate Occlusion
            # patch_size=40, stride=20 is a reasonable balance for 380x380 images
            heatmap, explained_class = occlusion(img_tensor, class_idx=predicted_label, patch_size=60, stride=30)
            
            # Overlay
            overlay, original = overlay_heatmap(img_tensor, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET)
            
            # Save
            filename = f"sample_{count:02d}_real_{real_label}_pred_{predicted_label}_conf_{confidence:.2f}.jpg"
            save_path = output_dir / filename
            
            # Plot
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            ax[0].imshow(original)
            ax[0].set_title(f"Original (Real: {real_label})")
            ax[0].axis('off')
            
            ax[1].imshow(heatmap, cmap='jet')
            ax[1].set_title("Occlusion Sensitivity")
            ax[1].axis('off')
            
            ax[2].imshow(overlay)
            ax[2].set_title(f"Overlay (Pred: {predicted_label})")
            ax[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved explanation to {save_path}")
            
            count += 1
        
        # if count >= max_count: break

    print("\nDone! Please check the 'explanation_occlusion' folder.")

if __name__ == "__main__":
    run_occlusion()
