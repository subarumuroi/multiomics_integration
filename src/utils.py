"""Utilities: IO, results directory, consensus feature identification."""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def create_results_dir(base_dir: str, name: str = "results") -> Path:
    """Create a results directory, return its path."""
    p = Path(base_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df, path):
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def save_json(data, path):
    """Save dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)


def find_consensus_features(importance_dfs: dict, top_n: int = 15) -> pd.DataFrame:
    """Identify features appearing across multiple methods' top-N lists.
    
    Uses rank-based ensemble feature selection: each method contributes its
    top-N features, and consensus is determined by vote count.  This avoids
    direct comparison of incompatible score magnitudes (VIP, Gini, coefficients)
    and instead asks whether a feature is robustly identified regardless of the
    analytic assumptions.
    
    Note: sPLS-DA and DIABLO share PLS lineage, so 4/4 agreement effectively
    represents 3 independent method families (PLS, tree-based, parametric linear)
    plus one multi-block variant.
    
    Parameters
    ----------
    importance_dfs : dict of {method_name: DataFrame with 'Feature' column}
        Each DataFrame should be sorted by importance (most important first).
    top_n : int
        Number of top features to consider per method.
    
    Returns
    -------
    DataFrame with Feature, n_methods, methods columns.
    """
    feature_methods = {}
    for method, df in importance_dfs.items():
        top_features = df.head(top_n)["Feature"].tolist()
        for f in top_features:
            if f not in feature_methods:
                feature_methods[f] = []
            feature_methods[f].append(method)
    
    records = [
        {"Feature": f, "n_methods": len(methods), "methods": ", ".join(methods)}
        for f, methods in feature_methods.items()
        if len(methods) > 1
    ]
    
    if not records:
        return pd.DataFrame(columns=["Feature", "n_methods", "methods"])
    
    df = pd.DataFrame(records).sort_values("n_methods", ascending=False).reset_index(drop=True)
    return df
