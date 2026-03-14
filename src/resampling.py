"""Resampling module using SMOTEENN for class imbalance handling.

This module applies SMOTEENN (SMOTE + Edited Nearest Neighbors)
to handle class imbalance in the dataset.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN


def show_class_distribution(
    y: pd.Series,
    title: str,
    output_dir: str,
    config: Dict[str, Any]
) -> None:
    """Display class distribution as bar plot and print counts.
    
    Args:
        y: Target variable.
        title: Plot title.
        output_dir: Directory to save plot.
        config: Configuration dictionary.
    
    Example:
        >>> show_class_distribution(y, "Before Resampling", output_dir, config)
    """
    logger = logging.getLogger(__name__)
    
    dpi = config.get("plots", {}).get("dpi", 300)
    fig_width = config.get("plots", {}).get("figure_width", 10)
    fig_height = config.get("plots", {}).get("figure_height", 6)
    
    counts = y.value_counts().sort_index()
    
    logger.info(f"Class distribution for '{title}':")
    for cls, count in counts.items():
        logger.info(f"  Class {cls}: {count} ({count/len(y)*100:.2f}%)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    classes = counts.index.astype(str)
    ax.bar(classes, counts.values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Severity Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    
    for i, (cls, count) in enumerate(counts.items()):
        ax.text(i, count + max(counts.values)*0.01, str(count), 
                ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    
    safe_title = title.lower().replace(" ", "_")
    output_path = output_dir / f"class_distribution_{safe_title}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Class distribution plot saved to {output_path}")


def apply_smoteenn(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTEENN to handle class imbalance.
    
    Automatically adjusts n_neighbors based on the smallest class count
    to avoid errors when classes have very few samples.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        config: Configuration dictionary.
    
    Returns:
        Tuple of (resampled features, resampled target).
    
    Example:
        >>> X_res, y_res = apply_smoteenn(X, y, config)
        >>> print(f"Resampled: {X_res.shape}")
    """
    logger = logging.getLogger(__name__)
    random_state = config.get("random_state", 42)
    
    # Calculate minimum class count to set appropriate n_neighbors
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    
    # n_neighbors must be less than the smallest class count
    # Default is 5, but we need to reduce it if classes are too small
    n_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
    
    logger.info(f"Class counts: {class_counts.to_dict()}")
    logger.info(f"Minimum class count: {min_class_count}, using n_neighbors={n_neighbors}")
    
    logger.info("Applying SMOTEENN...")
    
    # Import SMOTE to pass as parameter to SMOTEENN
    from imblearn.over_sampling import SMOTE
    
    smoteenn = SMOTEENN(
        smote=SMOTE(k_neighbors=n_neighbors),
        random_state=random_state,
        sampling_strategy="auto"
    )
    
    X_resampled, y_resampled = smoteenn.fit_resample(X, y)
    
    logger.info(
        f"SMOTEENN complete: {len(y)} -> {len(y_resampled)} samples "
        f"({len(y_resampled)/len(y)*100:.1f}%)"
    )
    
    return X_resampled, y_resampled


def compare_distributions(
    y_before: pd.Series,
    y_after: pd.Series,
    output_dir: str,
    config: Dict[str, Any]
) -> None:
    """Generate before/after class distribution comparison plot.
    
    Args:
        y_before: Target before resampling.
        y_after: Target after resampling.
        output_dir: Directory to save plot.
        config: Configuration dictionary.
    
    Example:
        >>> compare_distributions(y, y_res, output_dir, config)
    """
    logger = logging.getLogger(__name__)
    
    dpi = config.get("plots", {}).get("dpi", 300)
    fig_width = config.get("plots", {}).get("figure_width", 12)
    fig_height = config.get("plots", {}).get("figure_height", 6)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    counts_before = y_before.value_counts().sort_index()
    counts_after = y_after.value_counts().sort_index()
    
    all_classes = sorted(set(counts_before.index) | set(counts_after.index))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    x = np.arange(len(all_classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, 
                   [counts_before.get(c, 0) for c in all_classes], 
                   width, label="Before", color="steelblue", edgecolor="black")
    bars2 = ax.bar(x + width/2, 
                   [counts_after.get(c, 0) for c in all_classes], 
                   width, label="After", color="coral", edgecolor="black")
    
    ax.set_xlabel("Severity Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution: Before vs After SMOTEENN")
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes)
    ax.legend()
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height)}",
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha="center", va="bottom", fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    
    output_path = output_dir / "class_distribution_comparison.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Comparison plot saved to {output_path}")


def run_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Execute full resampling pipeline.
    
    Shows class distribution before, applies SMOTEENN,
    shows distribution after, and generates comparison plot.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        config: Configuration dictionary (should contain _project_root from preprocessing).
    
    Returns:
        Tuple of (resampled X, resampled y).
    
    Example:
        >>> X_res, y_res = run_resampling(X, y, config)
        >>> print(f"Resampled: {X_res.shape}")
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting resampling pipeline")
    
    # Get project_root from config (stored by preprocess_full_pipeline)
    project_root = config.get("_project_root", ".")
    output_dir = Path(project_root) / "outputs/figures"
    
    show_class_distribution(
        y, "Before SMOTEENN Resampling", str(output_dir), config
    )
    
    X_resampled, y_resampled = apply_smoteenn(X, y, config)
    
    show_class_distribution(
        y_resampled, "After SMOTEENN Resampling", str(output_dir), config
    )
    
    compare_distributions(y, y_resampled, str(output_dir), config)
    
    logger.info("Resampling pipeline complete")
    
    return X_resampled, y_resampled
