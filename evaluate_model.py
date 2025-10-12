"""
This module contains the main function for evaluating a trained model on test data.
It loads a model from checkpoint, runs inference on the test dataset, and saves
the classified images with naming convention XY_original_image_name where X is the
real class and Y is the predicted class.
Autor: Przemek Sekula
Created: 2025-01-25
Last modified: 2025-01-25
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import csv

from data_tools import ImageDataModule
from model import ImageClassifier
from tools import parse_with_config_file


def main(args):
    """
    Main function to evaluate the model on test data and save classified images.
    """
    print("--- Starting Model Evaluation ---")
    
    # Set up data module
    data_module = ImageDataModule(args)
    
    # Setup the data module to get the test dataset
    data_module.setup('test')
    
    # Construct checkpoint path using config
    checkpoint_path = os.path.join(args.checkpoint_dir, args.subdir_name, f"{args.model_name}-best-checkpoint.ckpt")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = ImageClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create CSV file for results
    csv_path = output_dir / "results.csv"
    
    # Get the test dataset
    test_dataset = data_module.test_dataset
    
    print(f"Evaluating on {len(test_dataset)} test images...")
    
    # Process each image in the test dataset
    correct_predictions = 0
    total_predictions = 0
    
    # Open CSV file for writing
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'real_class', 'predicted_class', 'predicted_probability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        with torch.no_grad():
            for idx, (image, true_label) in enumerate(test_dataset):
                # Add batch dimension and move to device
                image_batch = image.unsqueeze(0).to(device)
                
                # Get prediction
                logits = model(image_batch)
                predicted_label = torch.argmax(logits, dim=1).item()
                
                # Get predicted probability (softmax)
                probabilities = torch.softmax(logits, dim=1)
                predicted_probability = probabilities[0][predicted_label].item()
                
                # Get original image path and filename
                original_path, _ = test_dataset.file_list[idx]
                original_filename = original_path.name
                
                # Create new filename with format XY_original_name
                new_filename = f"{true_label}{predicted_label}_{original_filename}"
                
                # Load original image (without transforms) for saving
                original_image = Image.open(original_path).convert("RGB")
                
                # Save the image with new filename
                output_path = output_dir / new_filename
                original_image.save(output_path)
                
                # Write to CSV
                writer.writerow({
                    'image_name': original_filename,
                    'real_class': true_label,
                    'predicted_class': predicted_label,
                    'predicted_probability': f"{predicted_probability:.4f}"
                })
                
                # Track accuracy
                if true_label == predicted_label:
                    correct_predictions += 1
                total_predictions += 1
                
                # Print progress every 10 images
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(test_dataset)} images...")
    
    # Calculate and print final accuracy
    accuracy = correct_predictions / total_predictions
    print(f"\n--- Evaluation Complete ---")
    print(f"Total images processed: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classified images saved to: {output_dir.absolute()}")
    print(f"Results CSV saved to: {csv_path.absolute()}")


if __name__ == '__main__':
    pl.seed_everything(42)
    parser = argparse.ArgumentParser(
        description='Evaluates a trained model on test data. Defaults are loaded from configs.yaml, '
        'from the defaults section. To change the config, use the --configs argument.'
    )
    parser.add_argument("--configs", nargs="+", help="List of named configs from configs.yaml to load.")
    
    args = parse_with_config_file(parser, defaults_name="defaults")
    main(args)
