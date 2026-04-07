# BraTS_final

BraTS2020 segmentation project. This file is the only maintained top-level usage document.

## Runtime

Use the PyTorch Python from the local Anaconda/Miniconda environment:

```bash
/home/Creeken/miniconda3/envs/pytorch/bin/python
```

Recommended shell shortcut:

```bash
export PYTHON=/home/Creeken/miniconda3/envs/pytorch/bin/python
```

All commands below assume you run them from:

```bash
/home/Creeken/Desktop/machine-learning-test/BraTS_final
```

Examples below use `$PYTHON`. If you do not set it, replace `$PYTHON` with the full path above.

## What This Repo Covers

- Data preparation: convert BraTS2020 training data into `nnUNet_raw/Dataset220_BraTS2020`
- Training and validation: train the project 3D UNet and write fold artifacts
- Prediction and evaluation: sample training cases, run inference, evaluate, generate reports

Default project config:

- Dataset: `Dataset220_BraTS2020`
- Model: `BraTSFixed3DUNet`
- Configuration: `fixed_3d_fullres`
- Folds: `0 1 2 3 4`

## Key Paths

- Project root: [`/home/Creeken/Desktop/machine-learning-test/BraTS_final`](/home/Creeken/Desktop/machine-learning-test/BraTS_final)
- Raw dataset root: [`/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_raw`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_raw)
- Active raw dataset: [`/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_raw/Dataset220_BraTS2020`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_raw/Dataset220_BraTS2020)
- Active preprocessed root: [`/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_preprocessed`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_preprocessed)
- Training cases: [`/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_preprocessed/ProjectPlans_3d_fullres`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_preprocessed/ProjectPlans_3d_fullres)
- Ground truth: [`/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_preprocessed/gt_segmentations`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_preprocessed/gt_segmentations)
- Training outputs: [`/home/Creeken/Desktop/machine-learning-test/BraTS_final/training_results`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/training_results)
- Evaluation outputs: [`/home/Creeken/Desktop/machine-learning-test/BraTS_final/evaluation/results`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/evaluation/results)

Important: the preprocessed data is now flattened directly under `BraTS_final_Dataset/nnUNet_preprocessed/`. It is no longer expected under `BraTS_final_Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/`.

The code in [project.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/project.py) still accepts the legacy nested layout, but the current project data uses the flattened layout.

## Required Inputs

Before training or validation, these must exist under [`BraTS_final_Dataset/nnUNet_preprocessed`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_preprocessed):

- `dataset.json`
- `splits_final.json`
- `ProjectPlans_3d_fullres/`
- `gt_segmentations/`

Before report generation, the raw dataset must exist under [`BraTS_final_Dataset/nnUNet_raw/Dataset220_BraTS2020`](/home/Creeken/Desktop/machine-learning-test/BraTS_final/BraTS_final_Dataset/nnUNet_raw/Dataset220_BraTS2020) with at least:

- `imagesTr/`
- `labelsTr/`

## Entry Points

- [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/run.py): main CLI entry
- [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/cli.py): command definitions
- [project.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/project.py): dataset and path resolution

Main commands:

- `doctor`
- `train`
- `train-all`
- `validate`
- `find-best-config`
- `predict`
- `evaluate`
- `clean-last-results`

## First Check

Run this first to verify the project sees the correct paths:

```bash
$PYTHON run.py doctor
```

It prints:

- Project root
- Dataset name
- Model name
- Configuration
- Active preprocessed directory
- Training case directory
- GT directory

## Common Workflow

Full workflow:

```bash
$PYTHON run.py train-all --npz
$PYTHON run.py find-best-config
$PYTHON run.py predict --sample-training-cases 12 --sample-seed 123
$PYTHON run.py evaluate --pred-dir evaluation/results/predict_training_sample_n12_seed123_fold0
```

Single fold workflow:

```bash
$PYTHON run.py train --fold 0
$PYTHON run.py validate --fold 0
$PYTHON run.py evaluate --fold 0
```

Clean the latest auto-managed prediction/evaluation artifacts:

```bash
$PYTHON run.py clean-last-results
```

## Commands

Show command help:

```bash
$PYTHON run.py --help
```

### `doctor`

```bash
$PYTHON run.py doctor
```

Use this to confirm the resolved project paths before starting a long run.

### `train`

Train one fold. If a checkpoint exists, the trainer resumes automatically unless you force a restart.

```bash
$PYTHON run.py train --fold 0
$PYTHON run.py train --fold 0 --restart-training
$PYTHON run.py train --fold 0 --validation-only
$PYTHON run.py train --fold 0 --pretrained-weights /path/to/checkpoint.pth
$PYTHON run.py train --fold 0 --val-best --npz
```

Key args:

- `--fold`: required
- `--npz`: keep validation probabilities
- `--validation-only`: skip training and only run validation from a checkpoint
- `--restart-training`: force a fresh run for the fold
- `--pretrained-weights`: load external weights before training
- `--disable-checkpointing`: do not write checkpoints
- `--val-best`: prefer `checkpoint_best.pth` for validation

Outputs go to `training_results/fold<fold>/`.

### `train-all`

Run the default folds sequentially.

```bash
$PYTHON run.py train-all
$PYTHON run.py train-all --restart-training
$PYTHON run.py train-all --validation-only --val-best
$PYTHON run.py train-all --npz
```

### `validate`

Run validation for one fold using an existing checkpoint.

```bash
$PYTHON run.py validate --fold 0
$PYTHON run.py validate --fold 0 --val-best
$PYTHON run.py validate --fold 0 --npz
```

Outputs:

- Validation predictions: `training_results/fold<fold>/validation/`
- Validation summary: `training_results/fold<fold>/validation/summary.json`
- Fold summary copy: `training_results/fold<fold>/summary.json`

### `find-best-config`

Aggregate available validation results and write inference helper files.

```bash
$PYTHON run.py find-best-config
$PYTHON run.py find-best-config --search-root training_results
$PYTHON run.py find-best-config --search-root /abs/path/to/summary_parent
```

This writes:

- `inference_information.json`
- `inference_instructions.txt`
- `postprocessing.json`

### `predict`

Sample training cases, run prediction, then evaluate automatically.

```bash
$PYTHON run.py predict --sample-training-cases 12 --sample-seed 123
$PYTHON run.py predict --sample-training-cases 12 --sample-seed 123 --val-best --npz
$PYTHON run.py predict --sample-training-cases 12 --sample-seed 123 --output-dir evaluation/results/demo_predict --overwrite
```

Notes:

- `predict` does not expose `--fold`
- It auto-selects an available checkpoint
- If `find-best-config` already wrote `inference_information.json`, recommended folds are preferred
- It writes `sample_selection.json` and `summary.json` automatically

Key args:

- `--sample-training-cases`: required
- `--sample-seed`: required
- `--npz`: keep probability maps
- `--val-best`: prefer `checkpoint_best.pth`
- `--overwrite`: allow replacing an existing output directory
- `--output-dir`: custom prediction directory

### `evaluate`

Evaluate a validation directory or prediction directory and generate a report.

```bash
$PYTHON run.py evaluate
$PYTHON run.py evaluate --fold 0
$PYTHON run.py evaluate --pred-dir evaluation/results/demo_predict
```

Default behavior with no `--fold` and no `--pred-dir`:

- Find the most recent prediction directory under `evaluation/results/`
- Evaluate it
- Write a summary JSON
- Generate a report directory

Key args:

- `--fold`: use `training_results/fold<fold>/validation/`
- `--pred-dir`: explicit prediction directory
- `--gt-dir`: custom GT directory
- `--output-file`: custom summary output path
- `--raw-dataset-dir`: raw data directory used for the report
- `--report-output-dir`: custom report output directory
- `--sample-selection-file`: attach a sample list to the report
- `--num-processes`: metric worker count
- `--chill`: allow partial predictions instead of a full set

### `clean-last-results`

Remove the latest auto-managed prediction/evaluation outputs.

```bash
$PYTHON run.py clean-last-results
```

This does not remove fold checkpoints or the main validation outputs.

## Standalone Scripts

### First Case Visualization

Quick visualization of the first BraTS case found in the archive.

```bash
$PYTHON first_case_visualization/visualize_first_case.py
$PYTHON first_case_visualization/visualize_first_case.py \
  --data-root BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --output-dir first_case_visualization/output
```

Script:

- [visualize_first_case.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/first_case_visualization/visualize_first_case.py)

### Data Preparation

Convert the original BraTS2020 training set into the project raw dataset layout.

```bash
$PYTHON data_preparation/scripts/prepare_brats2020_for_project.py
$PYTHON data_preparation/scripts/prepare_brats2020_for_project.py \
  --src-root BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --project-raw BraTS_final_Dataset/nnUNet_raw
```

Force rebuild:

```bash
$PYTHON data_preparation/scripts/prepare_brats2020_for_project.py --force
```

Related files:

- [prepare_brats2020_for_project.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/scripts/prepare_brats2020_for_project.py)
- [data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/docs/data_contract.md)
- [raw_dataset.json](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/metadata/raw_dataset.json)

## Code Layout

- `training/src/core/`: runtime config and trainer
- `training/src/data/`: dataset, labels, transforms, data loading
- `training/src/models/`: model definitions
- `training/src/optimization/`: losses, optimizer, scheduler, pretrained loading
- `training/src/monitoring/`: logs and plots
- `validation/src/`: validation, prediction, inference, best-config selection
- `evaluation/src/`: metrics and report generation

## Notes

- All commands support `--help`
- Main generated outputs live under `training_results/`, `evaluation/results/`, and `first_case_visualization/output/`
- If the README and code disagree, trust the code and inspect [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/cli.py), [project.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/project.py), and `run.py --help` with the PyTorch Python above
