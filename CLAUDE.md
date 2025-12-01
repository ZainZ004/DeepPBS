# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepPBS is a PyTorch-based geometric deep learning research tool for predicting protein-DNA binding specificity from 3D structures. Published in Nature Methods, it uses graph neural networks on protein-DNA complex structures to predict DNA binding preferences.

## Key Commands

### Environment Setup
```bash
# Create conda environment (CUDA version)
conda env create -f deeppbs_linux.yml
conda activate deeppbs

# Or CPU-only version
conda env create -f deeppbs_linux_cpu_only.yml

# Install package in development mode
pip install -e .
```

### Preprocessing and Prediction Pipeline
```bash
cd run/process/
# Place PDB files in pdb/ directory
ls pdb > input.txt
./process_and_predict.sh  # Main pipeline - preprocesses structures and runs predictions
```

### Training
```bash
cd run/
# Configure data_dir and output_path in config.json first
./submit_cross.sh  # Launches 5-fold cross-validation training
```

### Interpretability Analysis
```bash
cd run/process/
./vis_interpret.sh <pdb_name_without_extension>  # Compute atom importance scores
# Requires PyMOL for visualization
```

### Single Structure Prediction
```bash
cd run/
python predict.py <path_to_npz_file> <output_dir> -c ./pred_configs/pred_config_deeppbs.json
```

## Architecture Overview

### Core Package Structure (`deeppbs/`)
- **`nn/`**: Neural network modules
  - `layers/`: Custom PyG layers (e.g., `EquiCoordGraphConv`)
  - `metrics/`: Evaluation metrics for PWM prediction
  - `utils/`: NN utilities and helpers
- **`_data/`**: Configuration files and standards (amino acid properties, nucleotide mappings)

### Workflow Scripts (`run/`)
- **`process/`**: Preprocessing pipeline
  - `process_co_crystal.py`: Main preprocessing script
  - `process_and_predict.sh`: Combined preprocessing + prediction
- **`models/`**: Pre-trained model storage
- **`output/`**: Prediction results

### Key Dependencies
- **PyTorch 2.3.0 + PyTorch Geometric 2.5**: Core deep learning framework
- **3DNA Suite**: DNA structural analysis (in `dependencies/bin/`)
- **Curves**: DNA curvature analysis
- **BioPython**: PDB parsing and manipulation
- **FreeSASA**: Solvent accessibility calculations

### Data Flow
1. **Input**: PDB/CIF files with protein-DNA complexes
2. **Preprocessing**: Extracts geometric features, generates meshes, calculates SASA/SESA
3. **Graph Construction**: Builds heterogeneous graphs with protein and DNA nodes
4. **Prediction**: Ensemble of geometric deep learning models predicts PWM
5. **Output**: Position weight matrices and binding specificity predictions

### Configuration Files
- **`run/config.json`**: Training parameters (data paths, model settings, optimizer)
- **`run/process/process_config.json`**: Preprocessing paths
- **`run/pred_configs/pred_config_deeppbs.json`**: Prediction model configuration
- **`run/interpret_configs/interpret_config_deeppbs.json`**: Interpretability settings

## Development Notes

### GPU Requirements
- CUDA 12.1/12.2 support required for GPU acceleration
- CPU-only mode available but significantly slower
- Docker container available with pre-configured environment

### Parallel Processing
- Preprocessing can be parallelized across multiple PDB files
- Each job should use separate working directory to avoid file conflicts
- SLURM scripts provided for cluster deployment

### Testing
No formal test suite exists - this is research code. Validation is done through:
- Cross-validation during training
- Benchmark dataset evaluation
- Manual inspection of predictions

### Common Issues
- **PyMOL dependency**: Required only for visualization, not core functionality
- **3DNA licensing**: Commercial use requires separate 3DNA license
- **Memory usage**: Large structures may require significant RAM during preprocessing