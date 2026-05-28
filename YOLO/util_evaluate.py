import argparse
import os
from alg_yolo import YOLOAlgorithm

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x_min, y_min, x_max, y_max] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Greedy matching of predicted boxes to ground truth boxes.
    :param gt_boxes: List of gt boxes [x1, y1, x2, y2].
    :param pred_boxes: List of tuples (box, conf) sorted descending by confidence.
    :return: (TPs, FPs, FNs)
    """
    tps = 0
    fps = 0
    matched_gt = set()
    
    # Sort predictions by confidence in descending order
    sorted_preds = sorted(pred_boxes, key=lambda x: x[1], reverse=True)
    
    for p_box, _ in sorted_preds:
        best_iou = -1.0
        best_gt_idx = -1
        
        for gt_idx, g_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_iou >= iou_threshold:
            tps += 1
            matched_gt.add(best_gt_idx)
        else:
            fps += 1
            
    fns = len(gt_boxes) - len(matched_gt)
    return tps, fps, fns

def compute_custom_metrics(model_path, test_image_paths, gt_annotations, iou_thresh=0.5, conf_thresh=0.25, device=None, save_visuals=True):
    """
    Calculate custom metrics (detection-only and normal classification) on test images.
    :param model_path: Path to trained YOLO .pt model checkpoint.
    :param test_image_paths: List of absolute paths to test images.
    :param gt_annotations: Dict mapping filename (basename) to its ground truth metadata:
                           {
                               'filename': str,
                               'boxes': [{'label': 'good'|'inflamed', 'box': [xtl, ytl, xbr, ybr]}]
                           }
    :param iou_thresh: IoU threshold for matching.
    :param conf_thresh: Confidence threshold for predictions.
    :param device: Device to run inference on (e.g. '0' or 'cpu').
    :param save_visuals: Whether to save prediction images with bounding boxes drawn.
    """
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO
    
    print(f"\n--- Running Custom Evaluation Metrics ---")
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # We will accumulate TP, FP, FN for:
    # 1. Detection-only
    det_tp, det_fp, det_fn = 0, 0, 0
    
    # 2. Classification per class (0: good, 1: inflamed)
    class_stats = {
        0: {'tp': 0, 'fp': 0, 'fn': 0},  # good
        1: {'tp': 0, 'fp': 0, 'fn': 0}   # inflamed
    }
    
    label_to_class = {'good': 0, 'inflamed': 1}
    
    print(f"Running predictions on {len(test_image_paths)} test images...")
    for idx, img_path in enumerate(test_image_paths):
        filename = os.path.basename(img_path)
        
        # Get ground truth boxes
        if filename not in gt_annotations:
            continue
            
        # Run prediction on a single image and optionally save visuals
        if save_visuals:
            from pathlib import Path
            save_dir = Path(model_path).parent.parent
            res = model.predict(
                source=img_path, 
                conf=conf_thresh, 
                verbose=False, 
                device=device, 
                save=True, 
                project=str(save_dir), 
                name='test_visual_predictions', 
                exist_ok=True
            )[0]
        else:
            res = model.predict(source=img_path, conf=conf_thresh, verbose=False, device=device)[0]
        
        gt_info = gt_annotations[filename]
        gt_boxes_with_labels = gt_info['boxes']
        
        # Get predictions
        pred_boxes_xyxy = res.boxes.xyxy.cpu().numpy().tolist()
        pred_classes = res.boxes.cls.cpu().numpy().tolist() # float (e.g. 0.0 or 1.0)
        pred_confs = res.boxes.conf.cpu().numpy().tolist()
        
        # 1. Match for Detection-Only
        # Ground truths (ignore class)
        gt_det_boxes = [b['box'] for b in gt_boxes_with_labels]
        # Predictions (ignore class)
        pred_det_boxes = [(box, conf) for box, conf in zip(pred_boxes_xyxy, pred_confs)]
        
        tp, fp, fn = match_boxes(gt_det_boxes, pred_det_boxes, iou_threshold=iou_thresh)
        det_tp += tp
        det_fp += fp
        det_fn += fn
        
        # 2. Match for Normal Classification (class-wise)
        for class_idx in [0, 1]:
            # Ground truths of class_idx
            gt_class_boxes = [b['box'] for b in gt_boxes_with_labels if label_to_class.get(b['label'], 0) == class_idx]
            # Predictions of class_idx
            pred_class_boxes = [
                (box, conf) for box, cls_val, conf in zip(pred_boxes_xyxy, pred_classes, pred_confs)
                if int(round(cls_val)) == class_idx
            ]
            
            tp_c, fp_c, fn_c = match_boxes(gt_class_boxes, pred_class_boxes, iou_threshold=iou_thresh)
            class_stats[class_idx]['tp'] += tp_c
            class_stats[class_idx]['fp'] += fp_c
            class_stats[class_idx]['fn'] += fn_c
            
    # Calculate final metrics
    def safe_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
        
    det_precision, det_recall, det_f1 = safe_metrics(det_tp, det_fp, det_fn)
    good_precision, good_recall, good_f1 = safe_metrics(
        class_stats[0]['tp'], class_stats[0]['fp'], class_stats[0]['fn']
    )
    inf_precision, inf_recall, inf_f1 = safe_metrics(
        class_stats[1]['tp'], class_stats[1]['fp'], class_stats[1]['fn']
    )
    
    macro_precision = (good_precision + inf_precision) / 2.0
    macro_recall = (good_recall + inf_recall) / 2.0
    macro_f1 = (good_f1 + inf_f1) / 2.0
    
    # Format and print the results beautifully
    lines = []
    lines.append("=" * 60)
    lines.append(f"Custom Evaluation Metrics Summary (IoU threshold = {iou_thresh})")
    lines.append("=" * 60)
    lines.append(f"{'Metric Class/Type':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    lines.append("-" * 60)
    lines.append(f"{'1. DETECTION ONLY (Any Scar)':<30} | {det_precision:<10.4f} | {det_recall:<10.4f} | {det_f1:<10.4f}")
    lines.append(f"   (TP: {det_tp}, FP: {det_fp}, FN: {det_fn})")
    lines.append("-" * 60)
    lines.append(f"{'2. CLASSIFICATION (good)':<30} | {good_precision:<10.4f} | {good_recall:<10.4f} | {good_f1:<10.4f}")
    lines.append(f"   (TP: {class_stats[0]['tp']}, FP: {class_stats[0]['fp']}, FN: {class_stats[0]['fn']})")
    lines.append(f"{'2. CLASSIFICATION (inflamed)':<30} | {inf_precision:<10.4f} | {inf_recall:<10.4f} | {inf_f1:<10.4f}")
    lines.append(f"   (TP: {class_stats[1]['tp']}, FP: {class_stats[1]['fp']}, FN: {class_stats[1]['fn']})")
    lines.append("-" * 60)
    lines.append(f"{'Macro Average':<30} | {macro_precision:<10.4f} | {macro_recall:<10.4f} | {macro_f1:<10.4f}")
    lines.append("=" * 60)
    
    summary_text = "\n".join(lines)
    print(summary_text)
    
    return {
        'detection': {'precision': det_precision, 'recall': det_recall, 'f1': det_f1, 'tp': det_tp, 'fp': det_fp, 'fn': det_fn},
        'class_good': {'precision': good_precision, 'recall': good_recall, 'f1': good_f1, 'tp': class_stats[0]['tp'], 'fp': class_stats[0]['fp'], 'fn': class_stats[0]['fn']},
        'class_inflamed': {'precision': inf_precision, 'recall': inf_recall, 'f1': inf_f1, 'tp': class_stats[1]['tp'], 'fp': class_stats[1]['fp'], 'fn': class_stats[1]['fn']},
        'macro': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
        'summary_text': summary_text
    }

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
