from ultralytics import YOLO
import os

class YOLOAlgorithm:
    """
    A class to handle YOLO model training and inference for scar detection.
    """
    def __init__(self, model_variant='yolo11n.pt'):
        """
        Initialize the YOLO model.
        :param model_variant: The YOLO model variant to use (e.g., 'yolo11n.pt', 'yolo11s.pt').
        """
        self.model = YOLO(model_variant)

    def train(self, data_yaml_path, epochs=100, imgsz=640, batch=16, project='YOLO_runs', name='scar_detection'):
        """
        Train the YOLO model on a custom dataset.
        :param data_yaml_path: Path to the data.yaml file defining the dataset.
        :param epochs: Number of training epochs.
        :param imgsz: Image size for training.
        :param batch: Batch size.
        :param project: Directory to save results.
        :param name: Name of the experiment.
        """
        print(f"Starting training with dataset: {data_yaml_path}")
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=os.path.abspath(project),
            name=name
        )
        return results

    def run_inference(self, source, conf=0.25, save=True, project='YOLO_inference', name='results'):
        """
        Run inference on images or videos.
        :param source: Path to image, folder, or video.
        :param conf: Confidence threshold.
        :param save: Whether to save the annotated results.
        :param project: Directory to save inference results.
        :param name: Name of the inference run.
        """
        print(f"Running inference on: {source}")
        results = self.model.predict(
            source=source,
            conf=conf,
            save=save,
            project=os.path.abspath(project),
            name=name
        )
        return results

    def validate(self, data_yaml_path, imgsz=640, project='YOLO/runs/scar_detection', name='evaluation'):
        """
        Validate the model on a dataset.
        :param data_yaml_path: Path to the data.yaml file.
        :param imgsz: Image size.
        :param project: Directory to save results.
        :param name: Name of the evaluation run.
        """
        print(f"Validating model on: {data_yaml_path}")
        results = self.model.val(
            data=data_yaml_path,
            imgsz=imgsz,
            project=os.path.abspath(project),
            name=name
        )
        return results

    def load_model(self, model_path):
        """
        Load a trained model checkpoint.
        :param model_path: Path to the .pt model file.
        """
        self.model = YOLO(model_path)
