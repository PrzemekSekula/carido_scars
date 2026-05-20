"""
Script to run Integrated Gradients explainability on the Trained Model.
Generates pixel-level attribution maps.

Autor: Przemek Sekula (Antigravity)
Created: 2026-01-20
"""

import sys
import os
from pathlib import Path

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ImageClassifier
from data_tools import ImageDataModule
from util_explainability import IntegratedGradients, visualize_ig
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

def grid_image(np_image):
    # If image is float, convert to [0, 255] uint8 for display
    if np_image.dtype == np.float32 or np_image.dtype == np.float64:
        if np_image.max() <= 1.0:
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = np_image.astype(np.uint8)
    return np_image

def run_ig():
    # 1. Setup Paths
    ckpt_path = r"d:\iitis\medical\carido_scars\checkpoints\All\tf_efficientnet_b4\tf_efficientnet_b4-best-checkpoint.ckpt"
    # Output path relative to this script's directory
    output_dir = Path(__file__).parent / "explanation_IG"
    output_dir.mkdir(exist_ok=True)
    
    # 2. Load Model
    print(f"Loading model from {ckpt_path}...")
    try:
        model = ImageClassifier.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    # 3. Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    
    # 4. Load Data
    config = Config()
    # Note: We are explicitly ignoring model.hparams.data_dir as per previous issues
    if hasattr(model.hparams, 'image_size'):
        config.image_size = model.hparams.image_size
        
    print(f"Loading data from {config.data_dir}...")
    dm = ImageDataModule(config)
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    
    # 5. Generate Explanations
    print("Generating IG explanations for all test samples...")
    
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
            
            # Generate IG
            # Use smaller steps if memory is tight, e.g., steps=20. Standard is 50.
            attributions, explained_class = ig(img_tensor, class_idx=predicted_label, steps=20)
            
            # Un-normalize original image for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            orig_img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            orig_img = std * orig_img + mean
            orig_img = np.clip(orig_img, 0, 1)
            
            # Visualize Attribution
            # visualize_ig expects attributions in (C, H, W) and original in (H, W, 3)
            heatmap = visualize_ig(attributions, orig_img)
            
            # Save
            filename = f"sample_{count:02d}_real_{real_label}_pred_{predicted_label}_conf_{confidence:.2f}.jpg"
            save_path = output_dir / filename
            
            # Plot
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            
            ax[0].imshow(grid_image(orig_img))
            ax[0].set_title(f"Original (Real: {real_label})")
            ax[0].axis('off')
            
            ax[1].imshow(heatmap)
            ax[1].set_title(f"Integrated Gradients (Pred: {predicted_label})")
            ax[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved explanation to {save_path}")
            
            count += 1
        
        # if count >= max_count: break

    print("\nDone! Please check the 'explanation_IG' folder.")

if __name__ == "__main__":
    run_ig()
