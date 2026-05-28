import os
import sys
import shutil
import random
# pyrefly: ignore [missing-import]
import torch
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

# Ensure SCRIPT_DIR is in path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from util_evaluate import compute_custom_metrics
from alg_yolo import YOLOAlgorithm

def parse_cvat_xml(xml_path):
    """
    Parse a CVAT XML annotation file.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return []
    
    annotations = []
    # CVAT XML can have <image> directly under <annotations> root
    for img_elem in root.findall('image'):
        img_name = img_elem.get('name')  # e.g., "Good/Leg/Leg_10_01.02.2025.jpeg"
        width = int(img_elem.get('width'))
        height = int(img_elem.get('height'))
        
        boxes = []
        for box_elem in img_elem.findall('box'):
            label = box_elem.get('label')  # "good" or "inflamed"
            xtl = float(box_elem.get('xtl'))
            ytl = float(box_elem.get('ytl'))
            xbr = float(box_elem.get('xbr'))
            ybr = float(box_elem.get('ybr'))
            boxes.append({
                'label': label,
                'box': [xtl, ytl, xbr, ybr]
            })
        annotations.append({
            'image_name': img_name,
            'width': width,
            'height': height,
            'boxes': boxes
        })
    return annotations

def find_actual_image_path(base_dir, name_in_xml):
    """
    Find actual image path on disk in a case-insensitive manner,
    supporting standard image extensions.
    """
    base_dir = Path(base_dir)
    # Check exact path
    exact_path = base_dir / name_in_xml
    if exact_path.exists():
        return exact_path
        
    # Check with normalized separators
    norm_name = name_in_xml.replace('/', os.sep).replace('\\', os.sep)
    norm_path = base_dir / norm_name
    if norm_path.exists():
        return norm_path
        
    # Try case-insensitive matching
    parent_dir = norm_path.parent
    if not parent_dir.exists():
        return None
        
    target_stem = norm_path.stem.lower()
    for f in parent_dir.iterdir():
        if f.is_file() and f.stem.lower() == target_stem:
            return f
            
    return None

def prepare_split(annotations, base_dir, test_ratio=0.15, val_ratio=0.15, seed=42):
    """
    Prepare stratified train-val-test split on parsed annotations.
    """
    random.seed(seed)
    
    # Separate into good and inflamed categories based on path or box label
    good_ann = []
    inflamed_ann = []
    
    for ann in annotations:
        # Check actual image existence
        img_path = find_actual_image_path(base_dir, ann['image_name'])
        if img_path is None:
            continue
            
        ann['resolved_path'] = img_path
        
        # Determine category: check if path contains 'inflamed' or 'good' (case insensitive)
        path_lower = str(img_path).lower()
        if 'inflamed' in path_lower:
            inflamed_ann.append(ann)
        else:
            good_ann.append(ann)
            
    print(f"Stratified Split Stats:")
    print(f"  Good category images: {len(good_ann)}")
    print(f"  Inflamed category images: {len(inflamed_ann)}")
    
    # Shuffle each group independently
    random.shuffle(good_ann)
    random.shuffle(inflamed_ann)
    
    # Split each group
    good_test_count = int(len(good_ann) * test_ratio)
    good_val_count = int(len(good_ann) * val_ratio)
    good_train = good_ann[good_test_count + good_val_count:]
    good_val = good_ann[good_test_count : good_test_count + good_val_count]
    good_test = good_ann[:good_test_count]
    
    inf_test_count = int(len(inflamed_ann) * test_ratio)
    inf_val_count = int(len(inflamed_ann) * val_ratio)
    inf_train = inflamed_ann[inf_test_count + inf_val_count:]
    inf_val = inflamed_ann[inf_test_count : inf_test_count + inf_val_count]
    inf_test = inflamed_ann[:inf_test_count]
    
    train_split = good_train + inf_train
    val_split = good_val + inf_val
    test_split = good_test + inf_test
    
    print(f"Final 3-Way Split size:")
    print(f"  Train: {len(train_split)} images")
    print(f"  Validation (val): {len(val_split)} images")
    print(f"  Test (holdout): {len(test_split)} images")
    
    return train_split, val_split, test_split

def generate_yolo_dataset(train_split, val_split, test_split, base_dir, output_dir):
    """
    Create YOLO dataset directories and copy files for train, val, and test splits.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"Cleaning existing temporary dataset directory: {output_dir}")
        shutil.rmtree(output_dir)
        
    # Create directories for train, val, and test subsets
    for subset in ['train', 'val', 'test']:
        (output_dir / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / subset).mkdir(parents=True, exist_ok=True)
        
    label_map = {'good': 0, 'inflamed': 1}
    
    def process_split(split_data, subset):
        copied_images = []
        for ann in split_data:
            src_img_path = ann['resolved_path']
            filename = src_img_path.name
            
            # Destination image path (keeping original filename)
            dest_img_path = output_dir / 'images' / subset / filename
            shutil.copy2(src_img_path, dest_img_path)
            copied_images.append(str(dest_img_path.resolve()))
            
            # Destination label path
            label_filename = src_img_path.stem + ".txt"
            dest_label_path = output_dir / 'labels' / subset / label_filename
            
            yolo_boxes = []
            img_w = ann['width']
            img_h = ann['height']
            
            for box_info in ann['boxes']:
                label = box_info['label'].lower()
                class_idx = label_map.get(label, 0)
                
                xtl, ytl, xbr, ybr = box_info['box']
                
                # Convert to normalized YOLO format
                x_center = (xtl + xbr) / (2.0 * img_w)
                y_center = (ytl + ybr) / (2.0 * img_h)
                w = (xbr - xtl) / img_w
                h = (ybr - ytl) / img_h
                
                # Clip values to [0.0, 1.0] just in case
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                yolo_boxes.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
            # Write labels to file (even if empty to be valid background image)
            with open(dest_label_path, 'w') as f:
                f.write("\n".join(yolo_boxes))
        return copied_images

    print(f"Generating train subset...")
    process_split(train_split, 'train')
    print(f"Generating validation (val) subset...")
    process_split(val_split, 'val')
    print(f"Generating test (holdout) subset...")
    test_image_paths = process_split(test_split, 'test')
    
    # Write dataset.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: good
  1: inflamed
"""
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"Dataset generated at {output_dir}")
    print(f"YOLO configuration written to {yaml_path}")
    return yaml_path, test_image_paths

def run_experiment(
    mode='train',
    model='yolo11n.pt',
    source=None,
    epochs=50,
    batch=16,
    imgsz=640,
    test_ratio=0.15,
    val_ratio=0.15,
    seed=42,
    conf=0.25,
    iou_thresh=0.5,
    project=None,
    name='hospital_experiment',
    device=None
):
    """
    Run YOLO train/evaluate experiment on hospital data.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if source is None:
        source = str(PROJECT_ROOT / 'data' / 'hospital_data')
    if project is None:
        project = str(PROJECT_ROOT / 'YOLO' / 'runs' / 'hospital_results')

    # Auto-detect CUDA GPU or CPU if not explicitly provided
    if device is None:
        try:
            import torch
            device = '0' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    print(f"🚀 Device Selected: {device} (CUDA matches: {device != 'cpu'})")

    base_dir = Path(source)
    if not base_dir.exists():
        print(f"Error: Hospital data directory '{source}' not found.")
        return
        
    # 1. Parse annotations from the 4 XML files
    annotations_dir = base_dir / 'annotations'
    xml_files = [
        annotations_dir / 'Good' / 'leg_annotations.xml',
        annotations_dir / 'Good' / 'stereotonomy_annotations.xml',
        annotations_dir / 'Inflamed' / 'leg_annotations.xml',
        annotations_dir / 'Inflamed' / 'stereotonomy_annotations.xml'
    ]
    
    print("Scanning for annotation files...")
    all_annotations = []
    for xml_file in xml_files:
        if xml_file.exists():
            print(f"  Parsing annotation: {xml_file}")
            parsed = parse_cvat_xml(xml_file)
            all_annotations.extend(parsed)
            print(f"    Found {len(parsed)} images annotated.")
        else:
            print(f"  Warning: Annotation file '{xml_file}' not found.")
            
    if not all_annotations:
        print("Error: No annotations parsed. Check your annotations folder structure.")
        return
        
    print(f"Total parsed images from XML annotations: {len(all_annotations)}")
    
    # 2. Perform parameterized stratified split
    train_split, val_split, test_split = prepare_split(
        all_annotations, base_dir, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
    )
    
    if not train_split or not val_split or not test_split:
        print("Error: Split resulted in empty train, val, or test sets.")
        return
        
    # 3. Generate YOLO formatted dataset
    yolo_dataset_dir = base_dir / 'yolo_dataset'
    yaml_path, test_image_paths = generate_yolo_dataset(
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        base_dir=base_dir,
        output_dir=yolo_dataset_dir
    )
    
    # Create absolute lookup of ground truths for test images (using filename as key)
    gt_annotations_lookup = {}
    for ann in test_split:
        filename = ann['resolved_path'].name
        gt_annotations_lookup[filename] = ann
        
    # Run based on mode
    if mode == 'train':
        print(f"\n--- Starting YOLO Training Mode ---")
        yolo_alg = YOLOAlgorithm(model_variant=model)
        
        # Train model
        train_results = yolo_alg.train(
            data_yaml_path=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device=device
        )
        
        # Best model path is automatically created by ultralytics inside the actual save folder
        try:
            actual_save_dir = Path(train_results.save_dir)
            results_dir = actual_save_dir
            print(f"YOLO training saved to actual directory: {results_dir}")
        except Exception:
            results_dir = Path(project) / name
            print(f"Warning: Could not get dynamic save_dir from train_results, using default: {results_dir}")
            
        best_model_path = results_dir / 'weights' / 'best.pt'
        if not best_model_path.exists():
            # Fallback to model variant if best.pt wasn't created
            print(f"Warning: Best model weight file not found at {best_model_path}. Using fallback {model}")
            best_model_path = Path(model)
    else:
        print(f"\n--- Starting YOLO Evaluation Mode ---")
        results_dir = Path(project) / name
        best_model_path = Path(model)
        if not best_model_path.exists() and not model.endswith('.pt'):
            print(f"Error: Model file '{model}' not found.")
            return

    # 4. Compute Custom Metrics using best model (with save_visuals=True)
    eval_results = compute_custom_metrics(
        model_path=str(best_model_path),
        test_image_paths=test_image_paths,
        gt_annotations=gt_annotations_lookup,
        iou_thresh=iou_thresh,
        conf_thresh=conf,
        device=device,
        save_visuals=True
    )
    
    # Save metrics summary to file
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / 'metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(eval_results['summary_text'])
        
    print(f"\n✅ Pipeline complete!")
    print(f"Custom evaluation metrics summary saved to: {summary_path}")

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    default_hospital_data = PROJECT_ROOT / 'data' / 'hospital_data'
    default_model = 'yolo11n.pt'
    default_project = PROJECT_ROOT / 'YOLO' / 'runs' / 'hospital_results'
    
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO on hospital data with custom metrics.")
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help="Mode: 'train' to train and evaluate, 'evaluate' to run evaluation on existing model.")
    
    parser.add_argument('--model', type=str, default=default_model,
                        help=f"Path to model or YOLO variant name (default: {default_model}).")
    
    parser.add_argument('--source', type=str, default=str(default_hospital_data),
                        help=f"Root path to hospital data (default: {default_hospital_data}).")
    
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs.")
    
    parser.add_argument('--batch', type=int, default=16,
                        help="Batch size for training.")
    
    parser.add_argument('--imgsz', type=int, default=640,
                        help="Image size for training/evaluation.")
    
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help="Proportion of data for test holdout split (default: 0.15).")
                        
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help="Proportion of data for training validation split (default: 0.15).")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for split (default: 42).")
                        
    parser.add_argument('--conf', type=float, default=0.25,
                        help="Confidence threshold for predictions.")
                        
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help="IoU threshold for box matching (default: 0.5).")
                        
    parser.add_argument('--project', type=str, default=str(default_project),
                        help=f"Directory to save results (default: {default_project}).")
                        
    parser.add_argument('--name', type=str, default='hospital_experiment',
                        help="Name of the run directory under project.")
                        
    parser.add_argument('--device', type=str, default=None,
                        help="Device to run on (e.g. '0' or 'cpu'). Default is None (auto-detect).")

    args = parser.parse_args()
    run_experiment(**vars(args))

if __name__ == "__main__":
    main()
