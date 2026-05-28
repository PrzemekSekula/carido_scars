"""
Script to run Grad-CAM explainability with UNIFIED sclaing on the Trained Model
This script loads the trained model and dataset, calculates the global maximum activation
across the test set, and generates Grad-CAM heatmaps using this unified scale.

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
from util_explainability import GradCAM, visualize_cam_with_colorbar

# Hardcoded config for convenience
class Config:
    data_dir = r"..\\Data\\Data" 
    batch_size = 1 # Keep it simple for analysis
    num_workers = 0
    test_split_size = 0.2
    val_split_size = 0.2
    image_size = [380, 380] # EfficientNet B4 size
    use_class_weights = False

def run_explainability_unified():
    # 1. Setup Paths
    ckpt_path = r"d:\iitis\medical\carido_scars\checkpoints\All\tf_efficientnet_b4\tf_efficientnet_b4-best-checkpoint.ckpt"
    # Output path relative to this script's directory
    output_dir = Path(__file__).parent / "explanation_gradCAM_unified"
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
    target_layer = model.backbone.conv_head
    print(f"Target layer for Grad-CAM: {target_layer}")
    
    # 4. Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 5. Load Data
    config = Config()
    if hasattr(model.hparams, 'image_size'):
        config.image_size = model.hparams.image_size
        
    print(f"Loading data from {config.data_dir}...")
    dm = ImageDataModule(config)
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    
    # 6. Pass 1: Determine Global Max for Unified Scaling
    print("Pass 1: calculating global maximum activation...")
    global_max = 0
    
    # We can limit the number of samples for speed if needed, but for 'unified scale' 
    # generally we want the whole set or a representative subset.
    # We'll stick to a max count to be safe/fast for this demo.
    max_samples = 150 # Process all test samples (dataset size is 118)
    processed_count = 0
    
    # Store samples to avoid re-loading/re-inference if memory allows
    # Each item: (img_tensor, real_label, pred_label, confidence, raw_cam, explained_class)
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
            
            # Generate Raw Grad-CAM (normalize=False)
            # We explain the PREDICTED class
            raw_cam, explained_class = grad_cam(img_tensor, class_idx=predicted_label, normalize=False)
            
            current_max = raw_cam.max()
            if current_max > global_max:
                global_max = current_max
            
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
            
    print(f"Global Maximum Activation found: {global_max:.4f}")
    
    # 7. Pass 2: Generate Visualizations with Unified Scale
    print("Pass 2: Generating unified-scale visualizations...")
    
    for item in tqdm(stored_results, desc="Saving Images"):
        img_tensor = item['img_tensor']
        raw_cam = item['raw_cam']
        real_label = item['real_label']
        pred_label = item['pred_label']
        conf = item['confidence']
        sample_id = item['id']
        
        # Use our new visualization function
        # We pass global_max as vmax. vmin is 0 (as ReLU is used).
        overlay, heatmap_colored, norm_heatmap = visualize_cam_with_colorbar(
            raw_cam, 
            img_tensor, 
            vmax=global_max, 
            vmin=0
        )
        
        # Save
        filename = f"sample_{sample_id:02d}_real_{real_label}_pred_{pred_label}_conf_{conf:.2f}.jpg"
        save_path = output_dir / filename
        
        # Create a composite image with matplotlib to include the colorbar
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        
        # 1. Original
        # We need to manually un-normalize img_tensor again for display here 
        # (visualize_cam_with_colorbar returned the overlay but duplicates logic for original)
        # Let's just use the helper's logic implicitly or do it manually.
        # Actually, visualize_cam_with_colorbar only returns the overlay and heatmap.
        # Let's assume we want to show the original too.
        
        # Quick un-norm for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        orig_img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        orig_img = std * orig_img + mean
        orig_img = np.clip(orig_img, 0, 1)
        
        ax[0].imshow(orig_img)
        ax[0].set_title(f"Original (Real: {real_label})")
        ax[0].axis('off')
        
        # 2. Heatmap with Scale
        im = ax[1].imshow(heatmap_colored) # This is just RGB, values are not helpful for colorbar
        # To show a colorbar that is accurate, we should plot the normalized heatmap data with a specific norm
        # But heatmap_colored is already processed by cv2 colormap.
        
        # Better approach for matplotlib consistency:
        # Plot the raw (or normalized) data using imshow and let matplotlib handle the colormap and bar
        # We use norm_heatmap which is [0, 1] relative to [0, global_max]
        # Or we use raw_cam and set vmin/vmax
        pcm = ax[1].imshow(raw_cam, cmap='jet', vmin=0, vmax=global_max)
        ax[1].set_title(f"Grad-CAM (Max: {raw_cam.max():.2f})")
        ax[1].axis('off')
        
        # Add colorbar to this axis
        fig.colorbar(pcm, ax=ax[1], fraction=0.046, pad=0.04)
        
        # 3. Overlay
        ax[2].imshow(overlay)
        ax[2].set_title(f"Overlay (Pred: {pred_label})")
        ax[2].axis('off')
        
        plt.suptitle(f"Unified Scale (0 - {global_max:.2f})", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    print(f"\nDone! Explanations saved to {output_dir}")

if __name__ == "__main__":
    run_explainability_unified()
