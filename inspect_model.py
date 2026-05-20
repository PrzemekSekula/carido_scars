import torch
from model import ImageClassifier
import timm

def inspect():
    # Path to the checkpoint
    ckpt_path = r"d:\iitis\medical\carido_scars\checkpoints\All\tf_efficientnet_b4\tf_efficientnet_b4-best-checkpoint.ckpt"
    
    try:
        # Load model
        print(f"Loading checkpoint from {ckpt_path}...")
        model = ImageClassifier.load_from_checkpoint(ckpt_path)
        model.eval()
        
        print(f"Model backbone: {model.hparams.model_name}")
        
        # Print last few layers of the backbone
        print("\n--- Backbone Layers (last 20) ---")
        for name, module in list(model.backbone.named_modules())[-20:]:
            print(name)
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect()
