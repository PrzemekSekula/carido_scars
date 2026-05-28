"""
Script to run Counterfactual Explainability on the Trained Model.
Generates counterfactual examples that flip the model's prediction.

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
from util_explainability import CounterfactualGenerator
import cv2

# Hardcoded config
class Config:
    data_dir = r"..\\Data\\Data" 
    batch_size = 1
    num_workers = 0
    test_split_size = 0.2
    val_split_size = 0.2
    image_size = [380, 380] # EfficientNet B4 size
    use_class_weights = False

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)

def run_counterfactual():
    # 1. Setup Paths
    ckpt_path = r"d:\iitis\medical\carido_scars\checkpoints\All\tf_efficientnet_b4\tf_efficientnet_b4-best-checkpoint.ckpt"
    # Output path relative to this script's directory
    output_dir = Path(__file__).parent / "explanation_counterfactual"
    output_dir.mkdir(exist_ok=True)
    
    # 2. Load Model
    print(f"Loading model from {ckpt_path}...")
    try:
        model = ImageClassifier.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    # 3. Initialize Generator
    cf_generator = CounterfactualGenerator(model)
    
    # 4. Load Data
    config = Config()
    if hasattr(model.hparams, 'image_size'):
        config.image_size = model.hparams.image_size
        
    print(f"Loading data from {config.data_dir}...")
    dm = ImageDataModule(config)
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    
    # 5. Generate Counterfactuals
    print("Generating Counterfactual Examples...")
    
    count = 0
    # Process only a subset or all? 
    # Counterfactual generation involves optimization loop, so it is slower.
    # We will process all but print progress often.
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        for i in range(images.size(0)):
            img_tensor = images[i:i+1]
            real_label = labels[i].item()
            
            # Predict original
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()
                conf_orig = probs[0, pred_label].item()
            
            # Generate CF (flip to other class)
            target_cls = 1 - pred_label
            cf_tensor, success, _ = cf_generator(img_tensor, target_class=target_cls, steps=100)
            
            # Get CF confidence
            with torch.no_grad():
                cf_logits = model(cf_tensor)
                cf_probs = torch.softmax(cf_logits, dim=1)
                cf_pred = torch.argmax(cf_probs, dim=1).item()
                cf_conf = cf_probs[0, cf_pred].item()
            
            status = "Success" if success else "Failed"
            
            # Calculate Difference Map (Absolute difference)
            # Use denormalized images for visual difference
            img_np = denormalize(img_tensor)
            cf_np = denormalize(cf_tensor)
            
            diff_map = np.abs(cf_np - img_np)
            # Enhance difference visibility: sum channels + normalize
            diff_vis = np.sum(diff_map, axis=2)
            diff_vis = (diff_vis - diff_vis.min()) / (diff_vis.max() - diff_vis.min() + 1e-8)
            diff_vis = np.uint8(255 * diff_vis)
            diff_heatmap = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
            diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)

            # Save
            filename = f"sample_{count:02d}_real_{real_label}_origPred_{pred_label}_cfPred_{cf_pred}_{status}.jpg"
            save_path = output_dir / filename
            
            # Plot
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            ax[0].imshow(img_np)
            ax[0].set_title(f"Original\nPred: {pred_label} ({conf_orig:.2f})")
            ax[0].axis('off')
            
            ax[1].imshow(cf_np)
            ax[1].set_title(f"Counterfactual ({status})\nPred: {cf_pred} ({cf_conf:.2f})")
            ax[1].axis('off')
            
            ax[2].imshow(diff_heatmap)
            ax[2].set_title("Difference Map (Changes)")
            ax[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"[{count}] {status}: Orig {pred_label}->{target_cls} | Saved to {save_path}")
            
            count += 1
            
    print("\nDone! Check 'explanation_counterfactual' folder.")

if __name__ == "__main__":
    run_counterfactual()
