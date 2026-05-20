import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def convert_paint_to_yolo(image_path, output_dir, color_rgb=[255, 0, 255], tolerance=10):
    """
    Finds pixels of a specific color and creates a YOLO bounding box around them.
    :param image_path: Path to the painted image.
    :param output_dir: Where to save the .txt label.
    :param color_rgb: The RGB color used for annotation (default: Magenta).
    :param tolerance: Tolerance for color matching.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a mask for the specific color
    lower = np.array([max(0, c - tolerance) for c in color_rgb])
    upper = np.array([min(255, c + tolerance) for c in color_rgb])
    mask = cv2.inRange(img_rgb, lower, upper)

    # Find connected components (multiple scars)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    yolo_labels = []
    h, w = img.shape[:2]

    # stats: [left, top, width, height, area]
    for i in range(1, num_labels):  # Skip background (0)
        x_min, y_min, width, height, area = stats[i]
        
        # Ignore very small scribbles (noise)
        if area < 20:
            continue

        # Convert to YOLO format (center_x, center_y, width, height) normalized [0, 1]
        center_x = (x_min + width / 2) / w
        center_y = (y_min + height / 2) / h
        norm_width = width / w
        norm_height = height / h

        # Class 0 for "scar"
        yolo_labels.append(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")

    if yolo_labels:
        label_filename = Path(image_path).stem + ".txt"
        with open(os.path.join(output_dir, label_filename), 'w') as f:
            f.write("\n".join(yolo_labels))
        print(f"Created labels for: {image_path} ({len(yolo_labels)} scars found)")
    else:
        print(f"No annotations found in: {image_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Paint-annotated images to YOLO labels.")
    parser.add_argument('--input', type=str, required=True, help="Path to folder containing painted images.")
    parser.add_argument('--output', type=str, required=True, help="Path to folder where YOLO .txt labels should be saved.")
    parser.add_argument('--color', type=int, nargs=3, default=[255, 0, 255], help="RGB color used in Paint (default: 255 0 255).")
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_extensions)]
    
    print(f"Processing {len(files)} images from {args.input}...")
    for f in files:
        convert_paint_to_yolo(os.path.join(args.input, f), args.output, color_rgb=args.color)

if __name__ == "__main__":
    main()
