"""
Weighted Gene Co-expression Network Analysis (WGCNA) adapted for metabolomics/proteomics.

Builds a weighted correlation network between features, identifies co-abundance
modules, tests module-trait associations, and identifies hub features.

Reference: Langfelder & Horvath (2008) BMC Bioinformatics.

NOTE: WGCNA was designed for n >= 15-20. With small n, pairwise correlations
are noisy and module detection less stable. Results should be interpreted as
exploratory and cross-referenced with supervised methods.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Correlation & adjacency
# ---------------------------------------------------------------------------

def compute_correlation_matrix(X, method="pearson"):
    """Compute feature-feature correlation matrix.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    method : 'pearson' or 'spearman'
        Spearman is more robust for small n.
    
    Returns
    -------
    corr : ndarray (n_features, n_features) — correlation matrix
    """
    if method == "spearman":
        corr = np.corrcoef(pd.DataFrame(X).rank().values.T)
    else:
        corr = np.corrcoef(X.T)
    # Fix numerical issues
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)
    return corr


def pick_soft_threshold(X, powers=None, method="pearson"):
    """Determine soft-thresholding power for scale-free topology.
    
    Tests multiple powers and returns the one that best fits a scale-free
    topology (R² of log(k) vs log(p(k)) regression).
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    powers : list of int, optional
        Candidate powers to test. Default: [2, 3, ..., 20]
    method : str
        Correlation method.
    
    Returns
    -------
    dict with 'power', 'r_squared', 'all_results' (DataFrame)
    """
    if powers is None:
        powers = list(range(2, 21))
    
    corr = compute_correlation_matrix(X, method=method)
    results = []
    
    for beta in powers:
        adj = np.abs(corr) ** beta
        np.fill_diagonal(adj, 0)
        
        # Connectivity (degree) of each node
        k = adj.sum(axis=0)
        
        # Scale-free fit: regress log(p(k)) ~ log(k)
        # Bin the connectivity distribution
        k_pos = k[k > 0]
        if len(k_pos) < 5:
            results.append({"power": beta, "r_squared": 0.0, "mean_connectivity": k.mean()})
            continue
        
        # Histogram-based approach
        n_bins = max(5, min(20, len(k_pos) // 3))
        hist, bin_edges = np.histogram(k_pos, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Remove empty bins
        mask = hist > 0
        if mask.sum() < 3:
            results.append({"power": beta, "r_squared": 0.0, "mean_connectivity": k.mean()})
            continue
        
        log_k = np.log10(bin_centers[mask])
        log_pk = np.log10(hist[mask].astype(float))
        
        # Linear regression
        slope, intercept = np.polyfit(log_k, log_pk, 1)
        predicted = slope * log_k + intercept
        ss_res = np.sum((log_pk - predicted) ** 2)
        ss_tot = np.sum((log_pk - log_pk.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # For scale-free, we want negative slope and high R²
        r2_signed = -np.sign(slope) * r2
        
        results.append({
            "power": beta, 
            "r_squared": r2_signed, 
            "mean_connectivity": k.mean(),
        })
    
    df = pd.DataFrame(results)
    
    # Pick first power where R² > 0.8, or highest R²
    good = df[df["r_squared"] > 0.8]
    if len(good) > 0:
        best = good.iloc[0]
    else:
        best = df.loc[df["r_squared"].idxmax()]
    
    return {
        "power": int(best["power"]),
        "r_squared": best["r_squared"],
        "all_results": df,
    }


def compute_adjacency(X, power=6, method="pearson"):
    """Compute adjacency matrix using signed hybrid network.
    
    adjacency_ij = |cor(i,j)|^power
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    power : int — soft-thresholding power
    method : str — correlation method
    
    Returns
    -------
    adjacency : ndarray (n_features, n_features)
    """
    corr = compute_correlation_matrix(X, method=method)
    adj = np.abs(corr) ** power
    np.fill_diagonal(adj, 0)
    return adj


def compute_tom(adjacency):
    """Topological Overlap Matrix (TOM).
    
    TOM measures interconnectedness: features sharing many neighbors 
    have high TOM even if their direct correlation is moderate.
    
    TOM_ij = (l_ij + a_ij) / (min(k_i, k_j) + 1 - a_ij)
    where l_ij = sum_u(a_iu * a_uj), k_i = sum_u(a_iu)
    
    Parameters
    ----------
    adjacency : ndarray (p, p) — adjacency matrix
    
    Returns
    -------
    tom : ndarray (p, p) — TOM similarity matrix
    """
    p = adjacency.shape[0]
    k = adjacency.sum(axis=0)  # connectivity
    
    # l_ij = sum_u(a_iu * a_uj) = (A @ A)_ij
    L = adjacency @ adjacency
    
    # Vectorized TOM computation
    # min(k_i, k_j) for all pairs
    k_row = k[np.newaxis, :]  # (1, p)
    k_col = k[:, np.newaxis]  # (p, 1)
    k_min = np.minimum(k_row, k_col)
    
    numerator = L + adjacency
    denominator = k_min + 1 - adjacency
    
    # Avoid division by zero
    tom = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    np.fill_diagonal(tom, 1.0)
    return tom


# ---------------------------------------------------------------------------
# Module detection
# ---------------------------------------------------------------------------

def detect_modules(tom, feature_names=None, min_module_size=3, cut_height=None):
    """Identify feature modules via hierarchical clustering on 1 - TOM.
    
    Parameters
    ----------
    tom : ndarray (p, p) — TOM similarity matrix
    feature_names : list of str
    min_module_size : int — minimum features per module (smaller merged to nearest)
    cut_height : float, optional — if None, uses dynamic tree cut heuristic
    
    Returns
    -------
    module_assignments : DataFrame with Feature, Module columns
    linkage_matrix : scipy linkage matrix (for dendrogram)
    """
    p = tom.shape[0]
    names = feature_names or [f"f{i}" for i in range(p)]
    
    # Distance = 1 - TOM
    dist = 1 - tom
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 1)
    
    # Convert to condensed form for scipy
    dist_condensed = squareform(dist, checks=False)
    
    # Hierarchical clustering (average linkage, as in WGCNA)
    Z = linkage(dist_condensed, method="average")
    
    # Dynamic cut height: use a reasonable default
    if cut_height is None:
        # Adaptive: target ~10-30 modules for typical datasets
        # Use the 99th percentile for cohesive modules, lower for small p
        if p > 500:
            cut_height = np.percentile(Z[:, 2], 99)
        elif p > 100:
            cut_height = np.percentile(Z[:, 2], 95)
        else:
            cut_height = np.percentile(Z[:, 2], 90)
    
    labels = fcluster(Z, t=cut_height, criterion="distance")
    
    # Merge small modules into Module 0 ("grey" / unassigned)
    module_counts = pd.Series(labels).value_counts()
    small_modules = module_counts[module_counts < min_module_size].index
    labels = np.array([0 if lab in small_modules else lab for lab in labels])
    
    # Renumber modules sequentially (1, 2, 3, ...) with 0 = unassigned
    unique_mods = sorted(set(labels) - {0})
    remap = {0: 0}
    for i, m in enumerate(unique_mods, 1):
        remap[m] = i
    labels = np.array([remap[lab] for lab in labels])
    
    df = pd.DataFrame({"Feature": names, "Module": labels})
    return df, Z


# ---------------------------------------------------------------------------
# Module-trait associations
# ---------------------------------------------------------------------------

def module_trait_correlation(X, y_encoded, module_assignments, method="pearson"):
    """Test association between module eigengenes and trait (ripening stage).
    
    Module eigengene = first PC of features in that module.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y_encoded : ndarray of ordinal integers (e.g. 0, 1, 2)
    module_assignments : DataFrame with Feature, Module columns
    method : 'pearson' or 'spearman'
    
    Returns
    -------
    DataFrame with Module, n_features, correlation, p_value
    """
    modules = sorted(module_assignments["Module"].unique())
    results = []
    eigengenes = {}
    
    for mod in modules:
        if mod == 0:
            continue  # skip unassigned
        
        mod_features = module_assignments[module_assignments["Module"] == mod]["Feature"].tolist()
        mod_idx = [i for i, f in enumerate(module_assignments["Feature"]) 
                   if module_assignments.iloc[i]["Module"] == mod]
        
        if len(mod_idx) < 2:
            continue
        
        # Module eigengene: first PC
        X_mod = X[:, mod_idx]
        X_centered = X_mod - X_mod.mean(axis=0)
        
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            eigengene = U[:, 0] * S[0]
        except np.linalg.LinAlgError:
            continue
        
        eigengenes[mod] = eigengene
        
        # Correlate with trait
        if method == "spearman":
            r, p = spearmanr(eigengene, y_encoded)
        else:
            r, p = pearsonr(eigengene, y_encoded)
        
        results.append({
            "Module": mod,
            "N_Features": len(mod_idx),
            "Correlation": r,
            "P_Value": p,
            "Abs_Correlation": abs(r),
        })
    
    df = pd.DataFrame(results).sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)
    return df, eigengenes


# ---------------------------------------------------------------------------
# Hub features
# ---------------------------------------------------------------------------

def identify_hub_features(X, module_assignments, adjacency, y_encoded=None, top_n=5):
    """Identify hub features: high intramodular connectivity + trait correlation.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    module_assignments : DataFrame with Feature, Module
    adjacency : ndarray (p, p)
    y_encoded : ndarray, optional — if provided, compute feature-trait correlation
    top_n : int — hub features per module
    
    Returns
    -------
    DataFrame with Feature, Module, Intramodular_Connectivity, Trait_Correlation, Is_Hub
    """
    all_features = module_assignments["Feature"].tolist()
    records = []
    
    modules = sorted(module_assignments["Module"].unique())
    
    for mod in modules:
        if mod == 0:
            continue
        
        mod_mask = module_assignments["Module"] == mod
        mod_indices = np.where(mod_mask.values)[0]
        mod_features = module_assignments.loc[mod_mask, "Feature"].tolist()
        
        if len(mod_indices) < 2:
            continue
        
        # Intramodular connectivity: sum of adjacency within module
        adj_sub = adjacency[np.ix_(mod_indices, mod_indices)]
        k_in = adj_sub.sum(axis=0)  # connectivity within module
        k_in_norm = k_in / k_in.max() if k_in.max() > 0 else k_in
        
        for i, feat_idx in enumerate(mod_indices):
            record = {
                "Feature": all_features[feat_idx],
                "Module": mod,
                "Intramodular_Connectivity": k_in[i],
                "Normalized_Connectivity": k_in_norm[i],
            }
            
            if y_encoded is not None:
                r, p = pearsonr(X[:, feat_idx], y_encoded)
                record["Trait_Correlation"] = r
                record["Trait_P_Value"] = p
                # Gene significance (GS) in WGCNA parlance
                record["Gene_Significance"] = abs(r)
            
            records.append(record)
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Hub score: product of normalized connectivity and gene significance
    if y_encoded is not None and "Gene_Significance" in df.columns:
        df["Hub_Score"] = df["Normalized_Connectivity"] * df["Gene_Significance"]
    else:
        df["Hub_Score"] = df["Normalized_Connectivity"]
    
    # Mark top hubs per module
    df["Is_Hub"] = False
    for mod in df["Module"].unique():
        mod_mask = df["Module"] == mod
        mod_df = df.loc[mod_mask].nlargest(top_n, "Hub_Score")
        df.loc[mod_df.index, "Is_Hub"] = True
    
    return df.sort_values(["Module", "Hub_Score"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full WGCNA pipeline
# ---------------------------------------------------------------------------

def run_wgcna(X, y_encoded, feature_names=None, power=None, corr_method="spearman",
              min_module_size=None, top_n_hubs=5):
    """Run full WGCNA pipeline on a single omics block.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y_encoded : ndarray of ordinal integers (0, 1, 2)
    feature_names : list of str
    power : int, optional — soft-thresholding power (auto-detected if None)
    corr_method : 'pearson' or 'spearman' (spearman recommended for small n)
    min_module_size : int, optional — auto-scaled if None (max(3, p//50))
    top_n_hubs : int — hub features to flag per module
    
    Returns
    -------
    dict with keys:
        'power' : int — selected soft threshold power
        'modules' : DataFrame — feature to module assignments
        'module_trait' : DataFrame — module-trait correlations
        'hubs' : DataFrame — hub features with connectivity & trait correlation
        'adjacency' : ndarray — adjacency matrix
        'tom' : ndarray — TOM matrix
        'linkage' : ndarray — hierarchical clustering linkage
        'eigengenes' : dict — {module: eigengene array}
    """
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    p = X.shape[1]
    
    # Auto-scale min_module_size for large feature sets
    if min_module_size is None:
        min_module_size = max(3, p // 50)
        if p > 1000:
            min_module_size = max(20, p // 50)
    
    # 1. Pick soft threshold
    if power is None:
        threshold_result = pick_soft_threshold(X, method=corr_method)
        power = threshold_result["power"]
        print(f"    Selected soft-threshold power: {power} (R² = {threshold_result['r_squared']:.3f})")
    
    # 2. Compute adjacency and TOM
    adj = compute_adjacency(X, power=power, method=corr_method)
    tom = compute_tom(adj)
    
    # 3. Detect modules
    modules_df, Z = detect_modules(tom, feature_names=names, min_module_size=min_module_size)
    n_modules = len(set(modules_df["Module"]) - {0})
    n_unassigned = (modules_df["Module"] == 0).sum()
    print(f"    Detected {n_modules} modules ({n_unassigned} unassigned features)")
    
    # 4. Module-trait correlations
    mod_trait_df, eigengenes = module_trait_correlation(X, y_encoded, modules_df, method=corr_method)
    
    # 5. Hub features
    hubs_df = identify_hub_features(X, modules_df, adj, y_encoded=y_encoded, top_n=top_n_hubs)
    
    return {
        "power": power,
        "modules": modules_df,
        "module_trait": mod_trait_df,
        "hubs": hubs_df,
        "adjacency": adj,
        "tom": tom,
        "linkage": Z,
        "eigengenes": eigengenes,
    }
