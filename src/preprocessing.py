"""Data preprocessing module for crash severity prediction.

This module handles data loading, cleaning, encoding, and scaling
for the US Accidents dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder


def get_project_root(config_path: str) -> Path:
    """Find project root by locating config file.
    
    Args:
        config_path: Path to the configuration YAML file.
    
    Returns:
        Path object pointing to project root (config file's parent directory).
    
    Example:
        >>> root = get_project_root('../config.yaml')
        >>> print(root)
        /Users/pavanhebli/Personal/Projects/Ev Crash Project
    """
    config_path = Path(config_path)
    
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    return config_path.parent.resolve()


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configure logging for the module.
    
    Args:
        config: Configuration dictionary containing logging settings.
    
    Returns:
        Configured logger instance.
    """
    log_level = config.get("logging", {}).get("level", "INFO")
    log_format = config.get("logging", {}).get(
        "format", "%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Configuration dictionary.
    
    Example:
        >>> config = load_config("config.yaml")
        >>> print(config["random_state"])
        42
    """
    logger = logging.getLogger(__name__)
    path = Path(config_path)
    
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        DataFrame containing the loaded data.
    
    Example:
        >>> df = load_data("data/US_Accidents_March23.csv")
        >>> print(df.shape)
        (7700000, 46)
    """
    logger = logging.getLogger(__name__)
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(path)
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def drop_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Conditionally drop columns based on config.
    
    Uses safe access pattern - drops columns only if 'columns.to_drop'
    is present in config and not empty.
    
    Args:
        df: Input DataFrame.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with columns removed if specified in config.
    
    Example:
        >>> df = drop_columns(df, config)
        >>> print(df.columns)
    """
    logger = logging.getLogger(__name__)
    
    columns_to_drop = config.get("columns", {}).get("to_drop", [])
    
    if not columns_to_drop:
        logger.info("No columns specified for dropping in config")
        return df
    
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    missing_cols = [col for col in columns_to_drop if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Columns not found in data: {missing_cols}")
    
    if existing_cols:
        df = df.drop(columns=existing_cols)
        logger.info(f"Dropped columns: {existing_cols}")
    
    return df


def handle_missing_values(
    df: pd.DataFrame, 
    config: Dict[str, Any]
) -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Uses mode for categorical columns and median for numerical columns.
    Optionally uses config to specify column types.
    
    Args:
        df: Input DataFrame.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with missing values handled.
    
    Example:
        >>> df = handle_missing_values(df, config)
        >>> print(df.isnull().sum().sum())
        0
    """
    logger = logging.getLogger(__name__)
    
    config_cols = config.get("columns", {})
    categorical_cols = config_cols.get("categorical_one_hot", []) + \
                       config_cols.get("categorical_ordinal", [])
    numerical_cols = config_cols.get("numerical", [])
    
    if not categorical_cols:
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    
    if not numerical_cols:
        numerical_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
    
    missing_before = df.isnull().sum().sum()
    
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            logger.debug(f"Filled categorical '{col}' with mode: {mode_val}")
    
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"Filled numerical '{col}' with median: {median_val}")
    
    missing_after = df.isnull().sum().sum()
    logger.info(
        f"Missing values handled: {missing_before} -> {missing_after} "
        f"({missing_before - missing_after} filled)"
    )
    
    return df


def convert_booleans_to_int(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Convert boolean columns to integers (0/1).
    
    Args:
        df: Input DataFrame.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with boolean columns converted to int.
    """
    logger = logging.getLogger(__name__)
    
    config_cols = config.get("columns", {})
    boolean_cols = config_cols.get("boolean", [])
    
    if not boolean_cols:
        logger.info("No boolean columns specified in config")
        return df
    
    existing_bool_cols = [col for col in boolean_cols if col in df.columns]
    
    df[existing_bool_cols] = df[existing_bool_cols].astype(int)
    
    logger.info(f"Converted {len(existing_bool_cols)} boolean columns to int")
    
    return df


def fill_twilight_from_time(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing twilight column values based on Start_Time hour.
    
    Hours 7-19 (7 AM to 7 PM) -> 'Day'
    Hours 0-6, 20-23 (Night) -> 'Night'
    
    Args:
        df: Input DataFrame with Start_Time column.
    
    Returns:
        DataFrame with twilight NaNs filled.
    """
    logger = logging.getLogger(__name__)
    
    twilight_cols = [
        'Sunrise_Sunset', 
        'Civil_Twilight', 
        'Nautical_Twilight', 
        'Astronomical_Twilight'
    ]
    
    if 'Start_Time' not in df.columns:
        logger.warning("Start_Time column not found, skipping twilight fill")
        return df
    
    if df['Start_Time'].dtype == 'object':
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    
    hours = df['Start_Time'].dt.hour
    
    day_hours = list(range(7, 20))
    
    for col in twilight_cols:
        if col not in df.columns:
            continue
            
        mask = df[col].isnull()
        if not mask.any():
            continue
        
        df.loc[mask & hours.isin(day_hours), col] = 'Day'
        df.loc[mask & ~hours.isin(day_hours), col] = 'Night'
        
        filled_count = mask.sum()
        if filled_count > 0:
            logger.info(f"Filled {filled_count} missing values in '{col}' based on time")
    
    return df


def _auto_detect_encoding_types(
    df: pd.DataFrame, 
    cardinality_threshold: int = 2
) -> Tuple[List[str], List[str]]:
    """Auto-detect encoding type based on cardinality.
    
    Columns with cardinality <= threshold get one-hot encoding,
    columns with higher cardinality get label encoding.
    
    Args:
        df: Input DataFrame.
        cardinality_threshold: Maximum unique values for one-hot encoding.
    
    Returns:
        Tuple of (one_hot_cols, ordinal_cols).
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    one_hot_cols = []
    ordinal_cols = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= cardinality_threshold and df[col].unique()[0] in []:
            one_hot_cols.append(col)
        else:
            ordinal_cols.append(col)
    
    return one_hot_cols, ordinal_cols


def encode_categoricals(
    df: pd.DataFrame, 
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    """Encode categorical features using one-hot and/or label encoding.
    
    Uses config to determine which columns to one-hot encode vs label encode.
    If not specified in config, auto-detects based on cardinality (threshold: 10).
    
    Args:
        df: Input DataFrame.
        config: Configuration dictionary.
    
    Returns:
        Tuple of (encoded DataFrame, list of new feature names).
    
    Example:
        >>> df_encoded, new_features = encode_categoricals(df, config)
        >>> print(f"Added {len(new_features)} new features")
    """
    logger = logging.getLogger(__name__)
    
    config_cols = config.get("columns", {})
    one_hot_cols = config_cols.get("categorical_one_hot", [])
    ordinal_cols = config_cols.get("categorical_ordinal", [])
    
    if not one_hot_cols and not ordinal_cols:
        logger.info("No encoding types specified in config, auto-detecting...")
        one_hot_cols, ordinal_cols = _auto_detect_encoding_types(df)
        logger.info(
            f"Auto-detected: {len(one_hot_cols)} for one-hot, "
            f"{len(ordinal_cols)} for label encoding"
        )
    
    df_encoded = df.copy()
    new_features = []
    
    if one_hot_cols:
        existing_one_hot = [col for col in one_hot_cols if col in df.columns]
        if existing_one_hot:
            df_encoded = pd.get_dummies(
                df_encoded, columns=existing_one_hot, drop_first=True
            )
            new_features.extend([
                col for col in df_encoded.columns 
                if col not in df.columns
            ])
            logger.info(f"One-hot encoding added {len(existing_one_hot)} columns")
    
    if ordinal_cols:
        existing_ordinal = [col for col in ordinal_cols if col in df.columns]
        label_encoders = {}
        for col in existing_ordinal:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            new_features.append(col)
        logger.info(f"Label encoded {len(existing_ordinal)} columns")
    
    if not one_hot_cols and not ordinal_cols:
        logger.info("No categorical columns found for encoding")
    
    return df_encoded, new_features


def scale_features(
    df: pd.DataFrame, 
    config: Dict[str, Any],
    target_column: str
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler.
    
    StandardScaler standardizes features by removing the mean and scaling
    to unit variance: z = (x - mu) / sigma
    
    Args:
        df: Input DataFrame.
        config: Configuration dictionary.
        target_column: Name of the target column to exclude from scaling.
    
    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    
    Example:
        >>> df_scaled, scaler = scale_features(df, config, "Severity")
    """
    logger = logging.getLogger(__name__)
    
    config_cols = config.get("columns", {})
    numerical_cols = config_cols.get("numerical", [])
    
    if not numerical_cols:
        numerical_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
    
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    if not numerical_cols:
        logger.info("No numerical columns to scale")
        return df, None
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    logger.info(f"Scaled {len(numerical_cols)} numerical features")
    
    return df, scaler


def preprocess_full_pipeline(
    config_path: str = "config.yaml"
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Execute the full preprocessing pipeline.
    
    Loads data, handles missing values, encodes categoricals,
    scales features, and returns clean X, y ready for modeling.
    
    Args:
        config_path: Path to the configuration YAML file.
    
    Returns:
        Tuple of (features DataFrame, target Series, config dict).
    
    Example:
        >>> X, y, config = preprocess_full_pipeline()
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting full preprocessing pipeline")
    
    config = load_config(config_path)
    
    # Check if project_root already exists in config, otherwise compute and store it
    project_root = config.get("_project_root")
    if not project_root:
        project_root = get_project_root(config_path)
        config["_project_root"] = str(project_root)
    
    setup_logging(config)
    
    # Resolve data path using stored project root
    data_path = Path(project_root) / config.get("data", {}).get("raw_path", "data/US_Accidents_March23.csv")
    target_column = config.get("data", {}).get("target_column", "Severity")
    
    df = load_data(str(data_path))
    
    df = fill_twilight_from_time(df)
    
    df = drop_columns(df, config)
    
    df = convert_booleans_to_int(df, config)
    
    df = handle_missing_values(df, config)
    
    # Encode target to 0-indexed (XGBoost requires 0,1,2,3 not 1,2,3,4)
    le_target = LabelEncoder()
    target = pd.Series(le_target.fit_transform(df[target_column].values), name=target_column)
    config["_target_encoder"] = le_target  # Store for potential inverse transform later
    
    df = df.drop(columns=[target_column])
    
    df, _ = encode_categoricals(df, config)
    
    df, scaler = scale_features(df, config, target_column)
    
    unique, counts = np.unique(target, return_counts=True)
    dist_str = "\n".join([f"  Class {u}: {c}" for u, c in zip(unique, counts)])
    logger.info(
        f"Preprocessing complete: X shape {df.shape}, "
        f"target distribution:\n{dist_str}"
    )
    
    return df, target, config
