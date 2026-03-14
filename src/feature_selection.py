"""Feature selection module using XGBoost and Random Forest.

This module trains both XGBoost and Random Forest classifiers,
extracts feature importances, and identifies the top features
common to both models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any]
) -> Tuple[XGBClassifier, np.ndarray]:
    """Train XGBoost classifier and extract feature importances.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        config: Configuration dictionary.
    
    Returns:
        Tuple of (trained model, feature importances array).
    
    Example:
        >>> model, importance = train_xgboost(X, y, config)
        >>> print(f"Top feature importance: {importance.max():.4f}")
    """
    logger = logging.getLogger(__name__)
    random_state = config.get("random_state", 42)
    
    logger.info("Training XGBoost classifier...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0
    )
    
    model.fit(X, y)
    
    importance = model.feature_importances_
    logger.info("XGBoost trained. Feature importances extracted.")
    
    return model, importance


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any]
) -> Tuple[RandomForestClassifier, np.ndarray]:
    """Train Random Forest classifier and extract feature importances.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        config: Configuration dictionary.
    
    Returns:
        Tuple of (trained model, feature importances array).
    
    Example:
        >>> model, importance = train_random_forest(X, y, config)
        >>> print(f"Top feature importance: {importance.max():.4f}")
    """
    logger = logging.getLogger(__name__)
    random_state = config.get("random_state", 42)
    
    logger.info("Training Random Forest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    importance = model.feature_importances_
    logger.info("Random Forest trained. Feature importances extracted.")
    
    return model, importance


def get_common_top_features(
    xgb_importance: np.ndarray,
    rf_importance: np.ndarray,
    feature_names: List[str],
    top_n: int = 12
) -> List[str]:
    """Find top N features common to both XGBoost and RF rankings.
    
    Ranks features by importance for each model separately,
    then returns the intersection of top N features from both.
    
    Args:
        xgb_importance: Feature importances from XGBoost.
        rf_importance: Feature importances from Random Forest.
        feature_names: List of feature names.
        top_n: Number of top features to consider from each model.
    
    Returns:
        List of feature names common to both models' top N.
    
    Example:
        >>> common_features = get_common_top_features(xgb_imp, rf_imp, features, 12)
        >>> print(f"Common features: {len(common_features)}")
    """
    logger = logging.getLogger(__name__)
    
    xgb_ranking = np.argsort(xgb_importance)[::-1]
    rf_ranking = np.argsort(rf_importance)[::-1]
    
    xgb_top = set(xgb_ranking[:top_n])
    rf_top = set(rf_ranking[:top_n])
    
    common_indices = xgb_top.intersection(rf_top)
    common_features = [feature_names[i] for i in common_indices]
    
    logger.info(
        f"Found {len(common_features)} common features in top {top_n} "
        f"from both models"
    )
    
    return common_features


def plot_feature_importance(
    xgb_importance: np.ndarray,
    rf_importance: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    config: Dict[str, Any]
) -> None:
    """Generate side-by-side feature importance bar plots.
    
    Creates two bar plots showing top 15 features from XGBoost
    and Random Forest, saved at 300 DPI.
    
    Args:
        xgb_importance: Feature importances from XGBoost.
        rf_importance: Feature importances from Random Forest.
        feature_names: List of feature names.
        output_dir: Directory to save plots.
        config: Configuration dictionary.
    
    Example:
        >>> plot_feature_importance(xgb_imp, rf_imp, features, output_dir, config)
    """
    logger = logging.getLogger(__name__)
    
    dpi = config.get("plots", {}).get("dpi", 300)
    fig_width = config.get("plots", {}).get("figure_width", 12)
    fig_height = config.get("plots", {}).get("figure_height", 6)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_features = min(15, len(feature_names))
    
    xgb_top_idx = np.argsort(xgb_importance)[::-1][:n_features]
    rf_top_idx = np.argsort(rf_importance)[::-1][:n_features]
    
    fig, axes = plt.subplots(1, 2, figsize=(2 * fig_width, fig_height))
    
    xgb_sorted_imp = xgb_importance[xgb_top_idx][::-1]
    xgb_sorted_names = [feature_names[i] for i in xgb_top_idx][::-1]
    axes[0].barh(xgb_sorted_names, xgb_sorted_imp, color="steelblue")
    axes[0].set_xlabel("Feature Importance")
    axes[0].set_title("XGBoost Feature Importance (Top 15)")
    
    rf_sorted_imp = rf_importance[rf_top_idx][::-1]
    rf_sorted_names = [feature_names[i] for i in rf_top_idx][::-1]
    axes[1].barh(rf_sorted_names, rf_sorted_imp, color="forestgreen")
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title("Random Forest Feature Importance (Top 15)")
    
    plt.tight_layout()
    
    output_path = output_dir / "feature_importance_comparison.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Feature importance plot saved to {output_path}")


def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any]
) -> List[str]:
    """Execute full feature selection pipeline.
    
    Trains both XGBoost and Random Forest, extracts importances,
    finds common top features, and generates comparison plots.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        config: Configuration dictionary (should contain _project_root from preprocessing).
    
    Returns:
        List of selected feature names.
    
    Example:
        >>> selected = run_feature_selection(X, y, config)
        >>> print(f"Selected {len(selected)} features")
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting feature selection pipeline")
    
    # Get project_root from config (stored by preprocess_full_pipeline)
    project_root = config.get("_project_root", ".")
    output_dir = Path(project_root) / "outputs/figures"
    
    feature_names = X.columns.tolist()
    
    xgb_model, xgb_importance = train_xgboost(X, y, config)
    
    rf_model, rf_importance = train_random_forest(X, y, config)
    
    top_n = config.get("feature_selection", {}).get("top_n", 12)
    common_features = get_common_top_features(
        xgb_importance, rf_importance, feature_names, top_n
    )
    
    plot_feature_importance(
        xgb_importance, rf_importance, feature_names, str(output_dir), config
    )
    
    logger.info(f"Feature selection complete. Selected {len(common_features)} features")
    
    return common_features
