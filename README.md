# MMTM-based Multi-Modal Survival Analysis Framework

## 📦 Project Overview

This project implements a multi-modal learning framework utilizing the **MMTM (Multi-modal Transfer Module)** mechanism for feature fusion. The primary task is **survival analysis prediction using Cox regression**. It includes code for pre-training via transfer learning and final inference.

## 📁 Project Structure

- **`MMTM.py`**  
  Fusion model code implementing the MMTM architecture.

- **`pre_train.py`**  
  Code for performing transfer learning-based pre-training.

- **`run.py`**  
  The main file for final model training and Cox survival analysis.

- **`util.py`**  
  Utility functions and helper modules.

- **`data/`**  
  Contains datasets for various cancers: `brca`, `cesc`, `ucec`, `ov`.

- **`model/`**  
  Stores pre-trained model files.

- **`output/`**  
  Contains prediction results and evaluation metrics.

## ▶️ How to Run

### Step 1: Pretrain the Model

```bash
python pre_train.py brca
```

### Step 2: Run the Main Process

```bash
python run.py brca
```

## 💻 Environment

- **Python:** 3.10
- **PyTorch:** 2.7.1+cpu

You can create a conda environment for reproducibility:

```bash
conda create -n mmtm_env python=3.10
conda activate mmtm_env
pip install torch+cpu torchvision torchaudio 
```

> ℹ️ Please install other required packages as needed (e.g., numpy, pandas, lifelines, sklearn).
