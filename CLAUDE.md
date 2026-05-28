# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Cardio scars binary image classification. PyTorch Lightning + `timm` backbones (Xception, EfficientNet variants). Two-phase training: frozen-backbone warmup, then fine-tune the last N backbone blocks. Datasets live outside the repo at `../Data/<Leg|Stereotomy|All>/<class>/.../*.jpeg`.

## Repository layout

- `initial_experiments/` — original classification pipeline (training, evaluation, analysis). Considered the *initial* set of experiments; new work goes into sibling folders rather than back into this one.
- `initial_paper/` — LaTeX source and figures for the initial paper (gitignored).
- `XAI/` — explainability experiments (Grad-CAM, occlusion, IG, counterfactual). They reuse `model.py` / `data_tools.py` from `initial_experiments/` by adding that folder to `sys.path`.
- `YOLO/` — YOLO-based scar/wound detection experiments. Self-contained; uses `ultralytics`.
- `new_datasets/` — download/visualization scripts for additional datasets (AZH, RedScar, SurgWound).
- `logs/`, `checkpoints/`, `test_results/` — runtime outputs (gitignored). Created in the current working directory at runtime; all commands below assume you run from the repository root.

## Common commands

All commands assume the repository root as the working directory.

Training (loads `defaults` from `initial_experiments/configs.yaml`, then merges named configs left-to-right):
```bash
python initial_experiments/train_model.py --configs xception_leg
python initial_experiments/train_model.py --configs tf_efficientnet_b4_all
# any default key can also be overridden inline, e.g.:
python initial_experiments/train_model.py --configs xception_all --batch_size 16 --num_epochs 10
```

Evaluation (loads `checkpoints/<subdir_name>/<model_name>-best-checkpoint.ckpt`, writes `test_results/results.csv` + per-image `XY_<name>.jpeg`):
```bash
python initial_experiments/evaluate_model.py --configs xception_all
```

Analysis (both read `test_results/results.csv`):
```bash
python initial_experiments/analyze_results.py   # interactive, plt.show()
python initial_experiments/analysis.py          # writes plots into initial_paper/, prints LaTeX-friendly metrics
```

TensorBoard:
```bash
tensorboard --logdir logs/
```

No test suite, linter, or formatter is configured.

## Config system (`initial_experiments/tools.py::parse_with_config_file`)

- `initial_experiments/configs.yaml` has a `defaults` section plus named override sections (`xception_leg`, `xception_all`, `tf_efficientnet_b4_all`, …).
- `parse_with_config_file` reads `configs.yaml` next to the script being run (`Path(sys.argv[0]).parent / "configs.yaml"`), starts from `defaults`, recursively merges each `--configs` name in order, then turns every key into a CLI flag (types inferred from the default value via `guess_type`). So any default can be overridden on the CLI.
- `subdir_name` is the per-experiment slot under both `logs/` and `checkpoints/` — change it when starting a new experiment to avoid clobbering old runs. Paths in `configs.yaml` use Windows separators (`..\\Data\\Leg`); they work on Linux too because `pathlib` handles them, but if you add new entries prefer forward slashes.
- `num_workers: 0` is set for Windows compatibility — bump it on Linux for faster data loading.

## Architecture (initial experiments)

**`initial_experiments/data_tools.py::ImageDataModule`** — Lightning DataModule. `setup()` does `Path(data_dir).rglob('*.jpeg')` and uses `path.parent.parent.name` as the class label (so the class folder is the *grandparent* of each image). Performs a stratified 60/20/20 train/val/test split with fixed `random_state=42`. When `use_class_weights: True`, computes inverse-frequency weights from the train split and exposes them as `self.class_weights`; `train_model.py` calls `setup('fit')` before constructing the model so it can pass these weights into the loss.

**`initial_experiments/model.py::ImageClassifier`** — Lightning module. Backbone via `timm.create_model(model_name, pretrained=True, num_classes=0)` with a `nn.Linear(num_features, num_classes)` head. Two-phase training:
1. Warmup (`current_epoch < warmup_epochs`): backbone fully frozen, only head trained at `learning_rate`.
2. Fine-tune (at `on_train_epoch_start` when `epoch == warmup_epochs`): unfreezes the last `unfreeze_layers` children of the backbone, then calls `self.trainer.strategy.setup_optimizers(self.trainer)` to rebuild the optimizer with two param groups (head at `learning_rate`, unfrozen backbone at `finetune_lr`). The `is_finetuning` flag drives `configure_optimizers`.

Metrics (`torchmetrics`, multiclass): accuracy on train/val each step, plus precision/recall/F1 macro and balanced accuracy on test. `ckpt_path='best'` in `trainer.test()` always evaluates the checkpoint with the highest `val_acc`.

**`initial_experiments/train_model.py`** — wires logger, data module, model, `ModelCheckpoint(monitor='val_acc', mode='max')`, `EarlyStopping(monitor='val_loss', patience=5)`. Logs hyperparams to TensorBoard via `tools.log_args_to_tensorboard`.

**`initial_experiments/evaluate_model.py`** — loads the checkpoint produced by training, iterates the test dataset one image at a time (no batching), and writes per-prediction rows to `test_results/results.csv` with fields `image_name,real_class,predicted_class,predicted_probability`. The probability column is the *softmax score of the predicted class*, not of class 1 — `analysis.py` and `analyze_results.py` convert it via `np.where(pred==1, p, 1-p)` before computing ROC/AUC.

**`initial_experiments/tmp.py`, `initial_experiments/tmp_copy.py`** — scratch files, not part of the pipeline.

## Conventions worth knowing

- Class folder layout is `data_dir/<class>/<patient_or_subdir>/*.jpeg` (grandparent-as-label). Adding a flat `data_dir/<class>/*.jpeg` layout will silently produce wrong class names.
- The checkpoint filename pattern is hardcoded as `{model_name}-best-checkpoint.ckpt` in both `train_model.py` and `evaluate_model.py` — they must match.
- `data_dir`, `subdir_name`, and `model_name` together pin down every artifact location; changing `model_name` mid-experiment will break `evaluate_model.py`'s checkpoint lookup.
- XAI scripts (`XAI/exp_*.py`) extend `sys.path` with `../initial_experiments` so they can `from model import ImageClassifier` / `from data_tools import ImageDataModule`. Run them from the repository root.

## Commit Strategy

Use a short, prefixed commit message. Supported prefixes (match the existing `git log`):

- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — internal restructuring, no behavior change
- `docs:` — documentation only
- `research:` — exploratory experiments / research iterations
- `chore:` — chores, code cleaning, etc.

If necessary, a commit may also include a longer description below the short prefixed subject line — separate the subject and body with a blank line and use the body for context, rationale, or follow-up notes that don't fit in the one-line summary.

When committing automatically, do **not** add a `Co-Authored-By: Claude ...` trailer or any other Claude Code attribution — commit under the user's identity only.
