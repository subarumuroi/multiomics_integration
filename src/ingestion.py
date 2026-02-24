"""Ingestion module: load, impute, filter, normalize, align omics layers."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_omics(filepath: str, group_col: str = "Groups", sample_col: str = "Sample") -> pd.DataFrame:
    """Load a single omics CSV. Returns DataFrame with Sample as index, Groups preserved."""
    df = pd.read_csv(filepath)
    if sample_col in df.columns:
        df = df.set_index(sample_col)
    elif df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.set_index(df.columns[0])
        df.index.name = "Sample"
    return df


def load_all_layers(data_dir: str, layers: Optional[dict] = None) -> dict:
    """Load multiple omics layers from a directory.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory.
    layers : dict, optional
        Mapping of layer names to filenames. If None, uses default banana dataset layout.
    
    Returns
    -------
    dict of {layer_name: DataFrame}
    """
    data_dir = Path(data_dir)
    if layers is None:
        layers = {
            "central_carbon": "badata-metabolomics.csv",
            "amino_acids": "badata-amino-acids.csv",
            "aromatics": "badata-aromatics.csv",
            "proteomics": "badata-proteomics-imputed.csv",
        }
    blocks = {}
    for name, fname in layers.items():
        fpath = data_dir / fname
        if fpath.exists():
            blocks[name] = load_omics(str(fpath))
        else:
            print(f"Warning: {fpath} not found, skipping {name}")
    return blocks


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def drop_sparse_features(df: pd.DataFrame, threshold: float = 0.5, group_col: str = "Groups") -> pd.DataFrame:
    """Drop features where fraction of missing values exceeds threshold."""
    feature_cols = [c for c in df.columns if c != group_col]
    frac_missing = df[feature_cols].isna().mean()
    keep = frac_missing[frac_missing <= threshold].index.tolist()
    return df[[group_col] + keep] if group_col in df.columns else df[keep]


def impute_half_min(df: pd.DataFrame, group_col: str = "Groups") -> pd.DataFrame:
    """Impute missing values with half the minimum observed value per feature.
    
    MNAR-appropriate: assumes missingness is due to values below detection limit.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != group_col]
    for col in feature_cols:
        if df[col].isna().any():
            half_min = df[col].min(skipna=True) / 2
            if np.isnan(half_min) or half_min <= 0:
                half_min = 1.0  # fallback for all-NaN or all-negative columns
            df[col] = df[col].fillna(half_min)
    return df


def impute_group_median(df: pd.DataFrame, group_col: str = "Groups") -> pd.DataFrame:
    """Impute missing values with within-group median, then half-min fallback."""
    df = df.copy()
    feature_cols = [c for c in df.columns if c != group_col]
    for col in feature_cols:
        if df[col].isna().any():
            group_medians = df.groupby(group_col)[col].transform("median")
            df[col] = df[col].fillna(group_medians)
            # Remaining NaNs (entire group was NaN) — half-min fallback
            if df[col].isna().any():
                half_min = df[col].min(skipna=True) / 2
                if np.isnan(half_min) or half_min <= 0:
                    half_min = 1.0
                df[col] = df[col].fillna(half_min)
    return df


def log_transform(X: np.ndarray, base: str = "log") -> np.ndarray:
    """Log-transform with small offset to handle zeros and negatives.
    
    For each column containing non-positive values, shifts the ENTIRE column
    by |col_min| + 1 so all values become positive before log transform.
    
    Parameters
    ----------
    X : array
    base : str
        'log' for natural log, 'log2', 'log10'
    """
    col_min = X.min(axis=0)
    # Per-column offset: shift entire column when any value is non-positive
    offset = np.where(col_min <= 0, np.abs(col_min) + 1, 0)
    X_pos = X + offset[np.newaxis, :] + 1e-10
    if base == "log2":
        return np.log2(X_pos)
    elif base == "log10":
        return np.log10(X_pos)
    else:
        return np.log(X_pos)


def pareto_scale(X: np.ndarray) -> tuple:
    """Pareto scaling: center and divide by sqrt(std).
    
    Preserves more biological variance than standard scaling.
    Returns (X_scaled, means, sqrt_stds) for inverse transform.
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    sqrt_stds = np.sqrt(stds)
    X_scaled = (X - means) / sqrt_stds
    return X_scaled, means, sqrt_stds


def standard_scale(X: np.ndarray) -> tuple:
    """Standard (z-score) scaling. Returns (X_scaled, means, stds)."""
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    X_scaled = (X - means) / stds
    return X_scaled, means, stds


# ---------------------------------------------------------------------------
# Prepare single block
# ---------------------------------------------------------------------------

def prepare_block(df: pd.DataFrame,
                  group_col: str = "Groups",
                  drop_threshold: float = 0.5,
                  imputation: str = "half_min",
                  transform: str = "log",
                  scaling: str = "pareto") -> tuple:
    """Full preprocessing pipeline for one omics block.
    
    Parameters
    ----------
    df : DataFrame with Groups column and features.
    group_col : name of the class label column.
    drop_threshold : max fraction of NaN allowed per feature.
    imputation : 'half_min' or 'group_median'.
    transform : 'log', 'log2', 'log10', or None.
    scaling : 'pareto', 'standard', or None.
    
    Returns
    -------
    X : ndarray (n_samples, n_features)
    y : ndarray of string labels
    feature_names : list of str
    sample_names : list of str
    """
    # Extract labels
    y = df[group_col].values
    sample_names = df.index.tolist()
    
    # Drop sparse features
    df = drop_sparse_features(df, threshold=drop_threshold, group_col=group_col)
    
    # Impute
    if imputation == "group_median":
        df = impute_group_median(df, group_col=group_col)
    else:
        df = impute_half_min(df, group_col=group_col)
    
    # Extract numeric matrix
    feature_cols = [c for c in df.columns if c != group_col]
    X = df[feature_cols].values.astype(float)
    feature_names = feature_cols
    
    # Remove zero-variance features
    variances = X.var(axis=0)
    keep = variances > 0
    X = X[:, keep]
    feature_names = [f for f, k in zip(feature_names, keep) if k]
    
    # Transform
    if transform is not None:
        X = log_transform(X, base=transform)
    
    # Scale
    if scaling == "pareto":
        X, _, _ = pareto_scale(X)
    elif scaling == "standard":
        X, _, _ = standard_scale(X)
    
    return X, y, feature_names, sample_names


# ---------------------------------------------------------------------------
# Multi-block alignment
# ---------------------------------------------------------------------------

def align_blocks(blocks: dict, group_col: str = "Groups") -> tuple:
    """Align multiple omics blocks to common samples.
    
    Parameters
    ----------
    blocks : dict of {name: DataFrame}
    
    Returns
    -------
    aligned : dict of {name: DataFrame} with common samples, same order
    common_samples : list of common sample IDs
    """
    sample_sets = [set(df.index) for df in blocks.values()]
    common = sorted(set.intersection(*sample_sets))
    
    if len(common) == 0:
        raise ValueError("No common samples found across omics layers.")
    
    for name, df in blocks.items():
        n_dropped = len(df) - len(common)
        if n_dropped > 0:
            print(f"  {name}: dropped {n_dropped} samples not in common set")
    
    aligned = {name: df.loc[common] for name, df in blocks.items()}
    return aligned, common


def prepare_multiblock(blocks: dict, group_col: str = "Groups", **prep_kwargs) -> tuple:
    """Preprocess and align multiple omics blocks.
    
    Returns
    -------
    X_blocks : dict of {name: ndarray}
    y : ndarray (shared labels)
    feature_names : dict of {name: list}
    sample_names : list
    """
    aligned, common_samples = align_blocks(blocks, group_col=group_col)
    
    X_blocks = {}
    feature_names = {}
    y = None
    
    for name, df in aligned.items():
        X, y_block, fnames, _ = prepare_block(df, group_col=group_col, **prep_kwargs)
        X_blocks[name] = X
        feature_names[name] = fnames
        if y is None:
            y = y_block
    
    return X_blocks, y, feature_names, common_samples


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

ORDINAL_MAP = {"Green": 0, "Ripe": 1, "Overripe": 2}

def encode_ordinal(y: np.ndarray) -> np.ndarray:
    """Encode labels preserving biological order: Green=0, Ripe=1, Overripe=2."""
    return np.array([ORDINAL_MAP[label] for label in y])

def decode_ordinal(y_enc: np.ndarray) -> np.ndarray:
    """Decode ordinal integers back to string labels."""
    inv_map = {v: k for k, v in ORDINAL_MAP.items()}
    return np.array([inv_map[val] for val in y_enc])
