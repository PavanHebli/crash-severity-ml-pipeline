# Crash Severity Feature Analysis

A modular ML pipeline for crash severity prediction demonstrating feature selection using XGBoost and Random Forest, and class imbalance handling using SMOTEENN.

## Features

- **Feature Selection**: Uses ensemble agreement between XGBoost and Random Forest to identify top predictors
- **Class Imbalance Handling**: Applies SMOTEENN (SMOTE + Edited Nearest Neighbors) for robust resampling
- **Configurable Pipeline**: All settings managed via `config.yaml` - no code changes needed
- **Reproducibility**: Fixed random seed (42) throughout
- **Production-Ready**: Comprehensive logging, type hints, Google-style docstrings

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd crash-severity-feature-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

1. Visit: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
2. Download: `US_Accidents_March23.csv`
3. Place in: `data/US_Accidents_March23.csv`

### 4. Run the pipeline

```bash
cd notebooks
jupyter notebook analysis_walkthrough.ipynb
```

## Project Structure

```
crash-severity-feature-analysis/
├── config.yaml              # Configuration settings
├── requirements.txt         # Dependencies
├── data/
│   └── README.md           # Dataset instructions
├── src/
│   ├── preprocessing.py    # Data loading, cleaning, encoding, scaling
│   ├── feature_selection.py # XGBoost + RF feature importance
│   └── resampling.py       # SMOTEENN resampling
├── notebooks/
│   └── analysis_walkthrough.ipynb
├── outputs/
│   └── figures/            # Generated plots
└── README.md
```

## Design Decisions

### Why SMOTEENN over SMOTE alone?

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic samples by interpolating between existing minority class instances. However, this can introduce noise, especially near class boundaries. SMOTEENN adds a cleaning step using Edited Nearest Neighbours (ENN), which removes samples misclassified by their neighbors. This combination:

- Prevents overfitting on noisy synthetic samples
- Cleans borderline examples that could confuse the classifier
- Produces more robust decision boundaries

### Why both XGBoost and Random Forest for feature selection?

Using both models provides:

1. **Reduced selection bias**: Each algorithm has different strengths - XGBoost excels at capturing complex interactions through gradient boosting, while Random Forest is more robust to overfitting through bagging
2. **Ensemble agreement**: Features that rank highly in both models are more likely to be genuinely important
3. **Robustness**: Intersection-based selection is less sensitive to individual model quirks

### Why top 12 features?

The top 12 features are selected as the intersection of both models' top-ranked features (configurable via `config.yaml`). This number provides:

- Balance between interpretability and predictive power
- Enough features to capture meaningful patterns
- Few enough to avoid overfitting

### Why this dataset?

The US Accidents dataset serves as a publicly accessible surrogate for the original TxDOT CRIS (Crash Records Information System) data used in the IEEE ICMLA 2025 paper. It provides:

- Real-world crash data with meaningful severity labels
- Rich feature set including environmental and road conditions
- Public availability for reproducibility

## Usage

### Configuration

Edit `config.yaml` to customize:

- `random_state`: Seed for reproducibility
- `data.raw_path`: Path to dataset
- `columns.to_drop`: Columns to remove
- `columns.categorical_one_hot`: Columns for one-hot encoding
- `columns.categorical_ordinal`: Columns for label encoding
- `feature_selection.top_n`: Number of top features to select
- `plots.dpi`: Plot resolution

### Output

The pipeline generates:
- `feature_importance_comparison.png`: Side-by-side XGBoost and RF importance plots
- `class_distribution_before_smoteenn_resampling.png`: Original class distribution
- `class_distribution_after_smoteenn_resampling.png`: Rebalanced class distribution
- `class_distribution_comparison.png`: Before/after comparison

## License

This project is for educational purposes.

## Attribution

> This implementation is by Pavan Hebli, based on methodology from the co-authored IEEE ICMLA 2025 paper. The original paper used TxDOT CRIS data; this repo uses US Accidents (Kaggle) as a public surrogate.
