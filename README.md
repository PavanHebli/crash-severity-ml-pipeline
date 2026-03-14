# Crash Severity Prediction
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A modular ML pipeline for predicting crash severity, featuring ensemble-based feature selection and class imbalance handling on real-world accident data.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Explanation](#pipeline-explanation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Results](#results)
- [Author](#author)
- [License](#license)

---

## Project Overview

Crash severity prediction is a class imbalance problem by nature, the vast majority of real-world accidents result in no injury, which causes naive models to ignore the minority classes that matter most: severe and fatal crashes.

This project tackles that problem with a clean, reproducible two-stage pipeline:

1. **Feature Selection** — identify the most predictive features from a high-dimensional crash dataset using XGBoost and Random Forest
2. **Class Imbalance Handling** — rebalance the dataset using SMOTEENN before model training

The pipeline is fully configurable via YAML, modular by design, and documented for reproducibility. It is intended as a reusable foundation for any tabular crash severity classification task.

---

## Features

- Feature selection using XGBoost + Random Forest (ensemble agreement approach)
- Class imbalance handling with SMOTEENN (SMOTE + Edited Nearest Neighbors)
- YAML-based centralized configuration — no hardcoded paths or parameters
- Modular pipeline architecture — preprocessing, feature selection, and resampling are fully independent modules
- Google-style docstrings and type hints throughout
- Structured logging via Python `logging` module
- Reproducible outputs with fixed random seeds
- Notebook walkthrough for visual exploration and presentation

---

## Dataset

**Source:** [US Accidents Dataset — Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

**Size:** ~7.7 million records, 46 columns

**Target Variable:** `Severity` - a 4-class scale (1 = least severe, 4 = most severe), heavily imbalanced toward class 2

**Features include:** weather conditions, road type, time of day, visibility, temperature, wind speed, road infrastructure flags (junction, crossing, traffic signal, etc.)

### How to Download

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to the [US Accidents dataset page](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
3. Click **Download** and extract `US_Accidents_March23.csv`
4. Place the file in the `data/` directory

---

## Installation

```bash
# Clone the repo
git clone https://github.com/PavanHebli/crash-severity-prediction.git
cd crash-severity-prediction

# Create conda environment
conda create -n EV_CRSH_ENV python=3.10
conda activate EV_CRSH_ENV

# Install dependencies
pip install -r requirements.txt

# macOS only — XGBoost requires OpenMP
brew install libomp
```

---

## Usage

### Step 1 — Download the dataset
Follow the instructions in the [Dataset](#dataset) section and place `US_Accidents_March23.csv` in `data/`.

### Step 2 — (Optional) Configure settings
Open `config.yaml` to adjust the dataset path, number of top features, or column definitions. Defaults work out of the box.

### Step 3 — Run the notebook

```bash
cd crash-severity-prediction
jupyter notebook notebooks/analysis_walkthrough.ipynb
```

The notebook walks through all three pipeline stages with markdown explanations and inline visualizations.

### Quick Test with Sample Data

A 5,000-row sample is included for fast iteration without downloading the full dataset:

```yaml
# In config.yaml, change:
data:
  raw_path: "data/sample_5000.csv"
```

Revert to the full dataset path when ready for final results.

---

## Pipeline Explanation

```
Raw Data (US Accidents CSV)
        │
        ▼
┌─────────────────────┐
│    Preprocessing    │  Drop irrelevant columns, impute missing values,
│  preprocessing.py   │  encode categoricals, scale numerical features
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Feature Selection  │  Train XGBoost (gradient-based gain) +
│ feature_selection.py│  Random Forest (mean decrease impurity)
│                     │  → Select top 12 features agreed by both
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Resampling       │  SMOTE generates synthetic minority samples
│   resampling.py     │  ENN removes noisy boundary majority samples
│                     │  → Balanced dataset ready for model training
└─────────────────────┘
        │
        ▼
  Balanced Feature-Selected Dataset
  (Ready for downstream model training)
```

**Why XGBoost + Random Forest together?**
XGBoost measures importance by gradient-based gain — how much each feature reduces the loss function. Random Forest measures by mean decrease in impurity. Each has blind spots: XGBoost can overweight early-split features; RF can favor high-cardinality columns. Selecting the top 12 features that both models independently agree on eliminates single-model bias. Agreement between two fundamentally different algorithms is strong evidence of genuine predictive value.

**Why SMOTEENN over plain SMOTE?**
SMOTE alone can generate synthetic samples in ambiguous regions near class boundaries, introducing noise. ENN removes those noisy majority samples post-oversampling. The combination produces a cleaner, more balanced dataset — without distorting the original feature distributions.

---

## Configuration

All settings are centralized in `config.yaml`:

| Key | Description | Default |
|-----|-------------|---------|
| `random_state` | Global seed for reproducibility | `42` |
| `data.raw_path` | Path to input CSV | `data/US_Accidents_March23.csv` |
| `data.target_column` | Target column name | `Severity` |
| `feature_selection.top_n` | Number of features to select | `12` |
| `plots.dpi` | Output figure resolution | `300` |
| `logging.level` | Logging verbosity | `INFO` |

Column configuration (which columns to drop, encode, or scale) is also fully defined in `config.yaml` — no hardcoded column names in any source file.

**Safe access pattern used throughout:**
```python
columns_to_drop = config.get("columns", {}).get("to_drop", [])
```
Missing keys never raise errors — they fall back to safe defaults.

---

## Project Structure

```
crash-severity-prediction/
├── config.yaml                         # Centralized configuration
├── requirements.txt                    # Pinned dependencies
├── data/
│   ├── README.md                       # Dataset download instructions
│   └── sample_5000.csv                 # 5K-row sample for quick testing
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                # Data loading, cleaning, encoding, scaling
│   ├── feature_selection.py            # XGBoost + Random Forest feature selection
│   └── resampling.py                   # SMOTEENN resampling + distribution plots
├── notebooks/
│   └── analysis_walkthrough.ipynb      # Visual pipeline walkthrough
├── outputs/
│   └── figures/                        # Generated plots (auto-created at runtime)
└── README.md
```

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
pyyaml>=6.0
```

---

## Results

Running the full pipeline produces the following outputs in `outputs/figures/`:

**Feature Importance Plots**
- `feature_importance_xgb.png` — Top features ranked by XGBoost gradient-based gain
- `feature_importance_rf.png` — Top features ranked by Random Forest mean decrease in impurity

**Class Distribution Plots**
- `class_distribution_before.png` — Original imbalanced class counts
- `class_distribution_after.png` — Balanced class counts after SMOTEENN

**Console / Log Output**
- Names of the top 12 selected features
- Class counts before and after resampling
- Pipeline step timing via structured logging

---

## Author

**Pavan Hebli**
[GitHub](https://github.com/PavanHebli) · [LinkedIn](https://linkedin.com/in/pavanhebli) · [Kaggle](https://www.kaggle.com/pavanhebli)

---

## License

MIT License. See `LICENSE` for details.
