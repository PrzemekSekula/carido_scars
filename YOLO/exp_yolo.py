import argparse
import os
from alg_yolo import YOLOAlgorithm

def main():
    # Set default paths based on your project structure
    default_source = os.path.join('data', 'wounds', 'valid', 'images')
    default_data = os.path.join('data', 'wounds', 'data.yaml')
    
    parser = argparse.ArgumentParser(description="Run YOLO experiments for scar detection.")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate'],
                        help="Execution mode: 'train' for training, 'predict' for inference, or 'evaluate' for validation.")
    
    # Model parameters
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help="Path to model weight file or YOLO variant name (e.g., yolo11n.pt).")
    
    # Training parameters
    parser.add_argument('--data', type=str, default=default_data,
                        help=f"Path to data.yaml file (default: {default_data}).")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument('--batch', type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="Image size for training and inference.")
    
    # Inference parameters
    parser.add_argument('--source', type=str, default=default_source,
                        help=f"Path to image, folder, or video for inference (default: {default_source}).")
    parser.add_argument('--conf', type=float, default=0.25,
                        help="Confidence threshold for inference.")
    
    # Logging/Output parameters
    parser.add_argument('--project', type=str, default=os.path.join('YOLO', 'runs', 'scar_detection'),
                        help="Directory to save experiment results (default: YOLO/runs/scar_detection).")
    parser.add_argument('--name', type=str, default='surgwound_inference',
                        help="Name of the experiment run.")

    args = parser.parse_args()

    # Initialize the algorithm
    yolo_alg = YOLOAlgorithm(model_variant=args.model)

    if args.mode == 'train':
        # Automatically set run name to dataset name if using default name
        if args.name == 'surgwound_inference':
            dataset_name = os.path.basename(os.path.dirname(args.data))
            args.name = dataset_name if dataset_name else 'custom_dataset'

        if not os.path.exists(args.data):
            print(f"Error: Dataset config file '{args.data}' not found.")
            print("Please create a data.yaml file as described in YOLO/README.md")
            return
        
        print(f"--- Starting YOLO Training Mode ---")
        yolo_alg.train(
            data_yaml_path=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name
        )
    
    elif args.mode == 'predict':
        print(f"--- Starting YOLO Prediction Mode ---")
        print(f"Model: {args.model}")
        print(f"Source: {args.source}")
        
        yolo_alg.run_inference(
            source=args.source,
            conf=args.conf,
            save=True,
            project=args.project,
            name=args.name
        )
        print(f"\n✅ Results saved to {args.project}/{args.name}")

    elif args.mode == 'evaluate':
        print(f"--- Starting YOLO Evaluation Mode ---")
        if not os.path.exists(args.data):
            print(f"Error: Dataset config file '{args.data}' not found.")
            return

        yolo_alg.validate(
            data_yaml_path=args.data,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name + '_eval'
        )
        print(f"\n✅ Evaluation results saved to {args.project}/{args.name}_eval")

if __name__ == "__main__":
    main()
