"""
Script to run Occlusion Sensitivity explainability with UNIFIED scaling on the Trained Model
This script loads the trained model and dataset, calculates the global maximum activation
across the test set for occlusion sensitivity, and generates heatmaps using this unified scale.

Autor: Przemek Sekula (Antigravity)
Created: 2026-02-05
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
from tqdm import tqdm
from model import ImageClassifier
from data_tools import ImageDataModule
from util_explainability import OcclusionSensitivity, visualize_cam_with_colorbar

# Hardcoded config for convenience
class Config:
    data_dir = r"..\\Data\\Data" 
    batch_size = 1
    num_workers = 0
    test_split_size = 0.2
    val_split_size = 0.2
    image_size = [380, 380] # EfficientNet B4 size
    use_class_weights = False

def run_occlusion_unified():
    # 1. Setup Paths
    ckpt_path = r"d:\iitis\medical\carido_scars\checkpoints\All\tf_efficientnet_b4\tf_efficientnet_b4-best-checkpoint.ckpt"
    # Output path relative to this script's directory
    output_dir = Path(__file__).parent / "explanation_occlusion_unified"
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
    # Params
    patch_size = 60 # Bigger patch for 380x380
    stride = 30
    print(f"Initializing Occlusion Sensitivity with patch_size={patch_size}, stride={stride}...")
    occlusion = OcclusionSensitivity(model)
    
    # 4. Load Data
    config = Config()
    if hasattr(model.hparams, 'image_size'):
        config.image_size = model.hparams.image_size
        
    print(f"Loading data from {config.data_dir}...")
    dm = ImageDataModule(config)
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    
    # 5. Pass 1: Determine Global Max/Min for Unified Scaling
    print("Pass 1: calculating global maximum/minimum sensitivity...")
    global_max = -np.inf
    global_min = np.inf
    
    max_samples = 150 # Process all
    processed_count = 0
    
    # Store samples to avoid re-loading/re-inference
    stored_results = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Scanning Dataset")):
        for i in range(images.size(0)):
            if processed_count >= max_samples:
                break
                
            img_tensor = images[i:i+1] # (1, C, H, W)
            real_label = labels[i].item()
            
            # Predict
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                predicted_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_label].item()
            
            # Generate Raw Occlusion Heatmap (normalize=False)
            raw_cam, explained_class = occlusion(
                img_tensor, 
                class_idx=predicted_label, 
                patch_size=patch_size, 
                stride=stride,
                normalize=False
            )
            
            current_max = raw_cam.max()
            current_min = raw_cam.min()
            
            if current_max > global_max:
                global_max = current_max
            if current_min < global_min:
                global_min = current_min
            
            stored_results.append({
                'img_tensor': img_tensor,
                'real_label': real_label,
                'pred_label': predicted_label,
                'confidence': confidence,
                'raw_cam': raw_cam,
                'id': processed_count
            })
            
            processed_count += 1
        
        if processed_count >= max_samples:
            break
            
    print(f"Global Sensitivity Range found: [{global_min:.4f}, {global_max:.4f}]")
    
    # Occlusion values can be negative (if occlusion INCREASES confidence, i.e., occluded feature was confusing)
    # or positive (if occlusion DECREASES confidence, i.e., occluded feature was important).
    # Typically we care about positive values (feature importance).
    # Ideally we should center around 0 or just show the positive influence.
    # Let's align with Grad-CAM: Show 0 to Max (ReLU behavior) or Min to Max?
    # Usually "red" means important (positive diff). "Blue" means unimportant or confusing?
    # Let's fix the scale from 0 to Global Max for consistency with "Importance", 
    # ignoring negative values (or clipping them to 0) similar to ReLU in GradCAM.
    # OR we can show the full range.
    
    # Let's choose to visualize only POSITIVE contributions (feature importance) for now, 
    # as that's what Grad-CAM does.
    
    v_min = 0 # Ignore negative impact (things that make the model MORE confident when removed)
    v_max = global_max
    
    print(f"Setting visualization range to [{v_min}, {v_max}]")
    
    # 6. Pass 2: Generate Visualizations with Unified Scale
    print("Pass 2: Generating unified-scale visualizations...")
    
    for item in tqdm(stored_results, desc="Saving Images"):
        img_tensor = item['img_tensor']
        raw_cam = item['raw_cam']
        real_label = item['real_label']
        pred_label = item['pred_label']
        conf = item['confidence']
        sample_id = item['id']
        
        # Clip negatives for visualization if we only care about importance
        raw_cam = np.maximum(raw_cam, 0)
        
        # Use our new visualization function
        overlay, heatmap_colored, norm_heatmap = visualize_cam_with_colorbar(
            raw_cam, 
            img_tensor, 
            vmax=v_max, 
            vmin=v_min
        )
        
        # Save
        filename = f"sample_{sample_id:02d}_real_{real_label}_pred_{pred_label}_conf_{conf:.2f}.jpg"
        save_path = output_dir / filename
        
        # Create a composite image with matplotlib to include the colorbar
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        
        # 1. Original
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        orig_img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        orig_img = std * orig_img + mean
        orig_img = np.clip(orig_img, 0, 1)
        
        ax[0].imshow(orig_img)
        ax[0].set_title(f"Original (Real: {real_label})")
        ax[0].axis('off')
        
        # 2. Heatmap with Scale
        pcm = ax[1].imshow(raw_cam, cmap='jet', vmin=v_min, vmax=v_max)
        ax[1].set_title(f"Occlusion Sensitivity (Max: {raw_cam.max():.2f})")
        ax[1].axis('off')
        
        # Add colorbar to this axis
        fig.colorbar(pcm, ax=ax[1], fraction=0.046, pad=0.04)
        
        # 3. Overlay
        ax[2].imshow(overlay)
        ax[2].set_title(f"Overlay (Pred: {pred_label})")
        ax[2].axis('off')
        
        plt.suptitle(f"Unified Scale ({v_min} - {v_max:.2f})", fontsize=14)
        plt.tight_layout()
        # plt.savefig(save_path)
        # Force saving to avoid potential issues? No, standard savefig is fine.
        fig.savefig(save_path)
        plt.close(fig)
        
    print(f"\nDone! Explanations saved to {output_dir}")

if __name__ == "__main__":
    run_occlusion_unified()
