# Project State & Roadmap

## Modules & Architecture Status
- [x] **evaluation module (`evaluation/`)**: Decoupled into `metrics.py`, `visualization.py`, `reporting.py`, `classification_evaluation.py`, and `VAE_evalution.py`.
- [x] **model_src module (`model_src/`)**: Unified model definitions (`tree/`, `VAE/`, and `vae_pipeline.py`).
- [x] **experiment framework (`experiments/` & `run.py`)**: 
  - `experiments/base_experiment.py`: `BaseExperiment` ABC handling YAML config loading, auto-generated timestamped run folders, configuration snapshotting for 100% reproducibility, and dataset loading helpers.
  - `experiments/configs/`: YAML-driven configs per experiment (e.g. `experiment1.yaml`).
  - `experiments/run.py` & root `run.py`: Unified CLI entry point (`python3 run.py --exp experiment1`, `python3 run.py --list`).
  - `experiments/experiment1.py`: Refactored to subclass `BaseExperiment`.
- [ ] **preprocessing refactor**: Preprocessing code in root needs to be modularized and moved into `data_handling/`.

---

## Experiment Framework Usage

```bash
# 1. List all registered experiments
python3 run.py --list

# 2. Run an experiment with default YAML config
python3 run.py --exp experiment1

# 3. Run with a custom config
python3 run.py --exp experiment1 --config path/to/custom_config.yaml

# 4. Run without timestamped subfolder (outputs directly to results/experiment1/)
python3 run.py --exp experiment1 --no-timestamp
```

---

## Results Directory Structure
```text
results/
├── experiment1/
│   ├── run_2026-10-11_02-10-00/      # Timestamped run
│   │   ├── config.yaml               # Snapshot of config used for this run
│   │   ├── vae_diagnostics/          # Loss curves, t-SNE, R² reconstruction
│   │   ├── vae_gradient_boosting/    # ROC, PR, Confusion Matrix, metrics_summary.json
│   │   └── gradient_boosting/        # Baseline GB metrics and curves
```
