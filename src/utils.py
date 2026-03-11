"""Utilities: IO, results directory, consensus feature identification."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

LAYER_ORDER = ["central_carbon", "amino_acids", "aromatics", "proteomics"]


def create_results_dir(base_dir: str, name: str = "results") -> Path:
    """Create a results directory, return its path."""
    p = Path(base_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df, path, index=False):
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


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


def _infer_layer(methods_str: str) -> str:
    """Infer the omics layer from the consensus methods string."""
    for layer in LAYER_ORDER:
        if layer in str(methods_str):
            return layer
    return "unknown"


def integrate_wgcna_evidence(consensus_df: pd.DataFrame, results_dir) -> pd.DataFrame:
    """Attach per-layer WGCNA support to consensus features.

    Parameters
    ----------
    consensus_df : DataFrame
        Must contain Feature and methods columns.
    results_dir : str or Path
        Root results directory containing single_omics/<layer>/wgcna outputs.

    Returns
    -------
    DataFrame with WGCNA support columns added.
    """
    if consensus_df.empty:
        return consensus_df.copy()

    results_dir = Path(results_dir)
    enriched = consensus_df.copy()
    enriched["layer"] = enriched["methods"].apply(_infer_layer)

    for col in [
        "wgcna_supported", "wgcna_module", "wgcna_module_size",
        "wgcna_module_trait_correlation", "wgcna_module_trait_p_value",
        "wgcna_is_hub", "wgcna_hub_score",
    ]:
        enriched[col] = np.nan

    enriched["wgcna_supported"] = False
    enriched["wgcna_is_hub"] = False

    for layer in sorted(enriched["layer"].unique()):
        if layer == "unknown":
            continue

        wgcna_dir = results_dir / "single_omics" / layer / "wgcna"
        module_path = wgcna_dir / "module_assignments.csv"
        trait_path = wgcna_dir / "module_trait_correlations.csv"
        hub_path = wgcna_dir / "hub_features.csv"

        if not module_path.exists():
            continue

        modules = pd.read_csv(module_path)
        trait = pd.read_csv(trait_path) if trait_path.exists() else pd.DataFrame()
        hubs = pd.read_csv(hub_path) if hub_path.exists() else pd.DataFrame()

        layer_mask = enriched["layer"] == layer
        for idx, row in enriched.loc[layer_mask].iterrows():
            feat = row["Feature"]
            mod_row = modules[modules["Feature"] == feat]
            if mod_row.empty:
                continue

            module_id = int(mod_row.iloc[0]["Module"])
            enriched.at[idx, "wgcna_module"] = module_id

            if module_id == 0:
                continue

            enriched.at[idx, "wgcna_supported"] = True
            enriched.at[idx, "wgcna_module_size"] = int((modules["Module"] == module_id).sum())

            if not trait.empty:
                trait_row = trait[trait["Module"] == module_id]
                if not trait_row.empty:
                    enriched.at[idx, "wgcna_module_trait_correlation"] = trait_row.iloc[0]["Correlation"]
                    enriched.at[idx, "wgcna_module_trait_p_value"] = trait_row.iloc[0]["P_Value"]

            if not hubs.empty:
                hub_row = hubs[hubs["Feature"] == feat]
                if not hub_row.empty:
                    enriched.at[idx, "wgcna_is_hub"] = bool(hub_row.iloc[0].get("Is_Hub", False))
                    enriched.at[idx, "wgcna_hub_score"] = hub_row.iloc[0].get("Hub_Score", np.nan)

    enriched["wgcna_support_score"] = (
        enriched["wgcna_supported"].astype(int)
        + enriched["wgcna_is_hub"].astype(int)
    )
    enriched["integrated_evidence_score"] = (
        enriched["n_methods"] + enriched["wgcna_support_score"]
    )

    enriched = enriched.sort_values(
        ["integrated_evidence_score", "n_methods", "wgcna_is_hub", "wgcna_hub_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return enriched
