import os
import argparse
from pathlib import Path
from alg_yolo import YOLOAlgorithm

def collect_images(root_path):
    """
    Recursively finds all image files in the given root path.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    # Default paths
    default_hospital_data = os.path.join('data', 'hospital_data')
    default_model = os.path.join('YOLO', 'runs', 'scar_detection', 'wounds', 'weights', 'best.pt')
    
    parser = argparse.ArgumentParser(description="Run YOLO inference on hospital data for manual review.")
    
    parser.add_argument('--model', type=str, default=default_model,
                        help=f"Path to the trained .pt model file (default: {default_model}).")
    
    parser.add_argument('--source', type=str, default=default_hospital_data,
                        help=f"Root path to hospital data (default: {default_hospital_data}).")
    
    parser.add_argument('--conf', type=float, default=0.25,
                        help="Confidence threshold for detection.")
    
    parser.add_argument('--project', type=str, default=os.path.join('YOLO', 'runs', 'hospital_results'),
                        help="Directory to save the results.")
    
    parser.add_argument('--name', type=str, default='manual_review',
                        help="Name of the inference run.")

    args = parser.parse_args()

    # 1. Initialize and load the trained model
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Please train the model first or provide a valid path.")
        return

    print(f"Loading model: {args.model}")
    yolo_alg = YOLOAlgorithm(model_variant=args.model)

    # 2. Collect all images recursively
    print(f"Scanning for images in: {args.source}")
    images = collect_images(args.source)
    print(f"Found {len(images)} images total.")

    if not images:
        print("No images found to process.")
        return

    # 3. Run Inference
    # Note: We pass the list of images directly to YOLO. 
    # YOLO's predict method can handle a list of paths.
    print(f"Starting inference on {len(images)} images...")
    
    # We run in batches or just pass the whole list if it's not too large.
    # YOLOv8+ can handle a list of paths efficiently.
    yolo_alg.run_inference(
        source=images,
        conf=args.conf,
        save=True,
        project=args.project,
        name=args.name
    )

    print(f"\n✅ Inference complete!")
    print(f"Results are ready for manual review in: {os.path.join(args.project, args.name)}")

if __name__ == "__main__":
    main()
