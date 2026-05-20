import argparse
import os
from alg_yolo import YOLOAlgorithm

def main():
    # Default paths
    default_data = os.path.join('data', 'wounds', 'data.yaml')
    default_model = os.path.join('YOLO', 'runs', 'scar_detection', 'wounds', 'weights', 'best.pt')
    
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    
    parser.add_argument('--model', type=str, default=default_model,
                        help=f"Path to the trained .pt model file (default: {default_model}).")
    
    parser.add_argument('--data', type=str, default=default_data,
                        help=f"Path to the data.yaml file (default: {default_data}).")
    
    parser.add_argument('--imgsz', type=int, default=640,
                        help="Image size for evaluation.")
    
    parser.add_argument('--project', type=str, default=os.path.join('YOLO', 'runs', 'scar_detection'),
                        help="Directory to save evaluation results.")
    
    parser.add_argument('--name', type=str, default=None,
                        help="Name of the evaluation run (default: <dataset_name>_eval).")

    args = parser.parse_args()

    # Auto-set name if not provided
    if args.name is None:
        dataset_name = os.path.basename(os.path.dirname(args.data))
        args.name = f"{dataset_name}_eval" if dataset_name else "evaluation"

    # Initialize and load the trained model
    yolo_alg = YOLOAlgorithm(model_variant=args.model)

    # Run evaluation
    results = yolo_alg.validate(
        data_yaml_path=args.data,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name
    )

    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"\n✅ Detailed results saved to: {os.path.join(args.project, args.name)}")

if __name__ == "__main__":
    main()
