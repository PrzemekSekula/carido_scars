# Cardio Scars

Binary image classification of cardiac surgical scars, plus follow-up experiments (explainability, YOLO-based detection, additional datasets). This is an ongoing research project; the code is not fully polished.

## Installation

1. Create a conda environment:
   ```bash
   conda create --name image_classification python=3.12.7
   conda activate image_classification
   ```

2. Install PyTorch following the official [instructions](https://pytorch.org/get-started/locally/) for your OS / CUDA version.

3. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository layout

- `initial_experiments/` — original classification pipeline (PyTorch Lightning + `timm`). Training, evaluation, and result analysis live here.
- `initial_paper/` — LaTeX source and figures for the initial paper (gitignored).
- `XAI/` — explainability experiments (Grad-CAM, occlusion, integrated gradients, counterfactual). They reuse modules from `initial_experiments/` via `sys.path`.
- `YOLO/` — YOLO-based scar/wound detection experiments.
- `new_datasets/` — download and visualization scripts for additional datasets (AZH, RedScar, SurgWound).

## Initial experiments

All commands are run from the repository root.

Train (configurations live in `initial_experiments/configs.yaml`):
```bash
python initial_experiments/train_model.py --configs xception_leg
python initial_experiments/train_model.py --configs tf_efficientnet_b4_all
```

Any default key can be overridden on the CLI:
```bash
python initial_experiments/train_model.py --configs xception_all --batch_size 16 --num_epochs 10
```

Evaluate a trained checkpoint and dump per-image predictions:
```bash
python initial_experiments/evaluate_model.py --configs xception_all
```

Analyze the predictions in `test_results/results.csv`:
```bash
python initial_experiments/analyze_results.py   # interactive
python initial_experiments/analysis.py          # writes plots into initial_paper/
```

TensorBoard:
```bash
tensorboard --logdir logs/
```

Datasets are expected at `../Data/<Leg|Stereotomy|All>/<class>/<patient_or_subdir>/*.jpeg` (the class folder must be the *grandparent* of each image).

## License

See [LICENSE](LICENSE).
