# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepPBS is a geometric deep learning framework for protein-DNA binding specificity prediction. It uses PyTorch and PyTorch Geometric to process protein-DNA complex structures and predict binding preferences as position weight matrices (PWMs).

## Common Development Commands

### Environment Setup
```bash
conda activate deeppbs_install  # Activate the conda environment
```

### Processing and Prediction Pipeline
```bash
cd run/process/
ls pdb > input.txt                    # Create input file from PDB directory
./process_and_predict.sh              # Process structures and make predictions
./vis_interpret.sh <pdb_name>         # Compute interpretability scores
```

### Training
```bash
cd run/
./submit_cross.sh                     # Run 5-fold cross-validation training
```

### Docker Usage
```bash
docker pull aricohen/deeppbs:latest
docker run --gpus all -it -v $(pwd)/test.cif:/app/input/test.cif -v $(pwd)/results:/output aricohen/deeppbs /app/input/test.cif
```

## Architecture Overview

The codebase follows a modular architecture with clear separation between data processing, neural network components, and execution scripts:

### Core Modules (`deeppbs/`)
- **`nn/`**: Neural network implementation
  - `layers/`: Custom geometric layers for protein-DNA graphs
  - `trainer.py`: Training loop and loss computation
  - `evaluator.py`: Model evaluation and metrics
  - `metrics/`: Custom metrics for binding specificity prediction
- **`_data/`**: JSON configuration files for model architectures and data processing

### Execution Pipeline (`run/`)
- **`process/`**: Data preprocessing pipeline
  - `process_co_crystal.py`: Main preprocessing script
  - `proc_source.sh`: Environment setup for bioinformatics tools
- **`models/`**: Model architecture definitions
- **`driver.py`: Main training orchestrator
- **`predict.py`: Inference script for pre-trained models

### Key Processing Flow
1. **Input**: PDB/mmCIF files with protein-DNA complexes
2. **Preprocessing**: Extract geometric features using 3DNA, Curves, APBS
3. **Graph Construction**: Convert to PyTorch Geometric graphs
4. **Prediction**: Ensemble of geometric neural networks
5. **Output**: PWMs and interpretability scores

### Configuration System
- **`run/config.json`**: Training configuration (data paths, hyperparameters)
- **`run/process/pred_configs/`**: Prediction-specific configurations
- **`deeppbs/_data/`**: Model architecture definitions

### External Dependencies
The project relies on several bioinformatics tools in `dependencies/bin/`:
- 3DNA: DNA structural analysis
- Curves: DNA curvature calculation
- APBS: Electrostatics calculations
- MSMS/NanoShaper: Molecular surface generation

These are sourced via `run/process/proc_source.sh` during processing.