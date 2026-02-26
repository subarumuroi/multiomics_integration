"""
sPLS-DA and DIABLO (multi-block sPLS-DA) — native Python implementation.

sPLS-DA: sparse Partial Least Squares Discriminant Analysis
DIABLO: Data Integration Analysis for Biomarker discovery using Latent cOmponents
         (multi-block extension of sPLS-DA)

Reference: 
  - Lê Cao et al. (2011) Bioinformatics — sPLS-DA
  - Singh et al. (2019) Bioinformatics — DIABLO

The algorithm:
  1. PLS on X vs dummy-encoded Y, with L1 (soft-thresholding) on loadings
  2. NIPALS deflation to extract successive components
  3. For DIABLO: maximize weighted covariance across blocks via design matrix
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Soft thresholding
# ---------------------------------------------------------------------------

def _soft_threshold(w: np.ndarray, keepX: int) -> np.ndarray:
    """Apply L1 penalty by keeping only the top keepX loadings by absolute value."""
    if keepX >= len(w):
        return w
    # Zero out all but top keepX
    abs_w = np.abs(w)
    threshold_idx = np.argsort(abs_w)[::-1]
    mask = np.zeros_like(w, dtype=bool)
    mask[threshold_idx[:keepX]] = True
    w_sparse = w * mask
    # Re-normalize
    norm = np.linalg.norm(w_sparse)
    if norm > 0:
        w_sparse = w_sparse / norm
    return w_sparse


# ---------------------------------------------------------------------------
# Single-block sPLS-DA
# ---------------------------------------------------------------------------

class SPLSDA:
    """Sparse PLS-DA: single-block discriminant analysis with feature selection.
    
    Parameters
    ----------
    n_components : int
        Number of latent components to extract.
    keepX : list of int or int
        Number of features to keep per component. If int, same for all components.
    max_iter : int
        Maximum NIPALS iterations per component.
    tol : float
        Convergence tolerance.
    """
    
    def __init__(self, n_components: int = 2, keepX=None, max_iter: int = 500, tol: float = 1e-6):
        self.n_components = n_components
        self.keepX = keepX
        self.max_iter = max_iter
        self.tol = tol
        
        # Fitted attributes
        self.x_weights_ = None      # W: (p, n_components)
        self.x_loadings_ = None     # P: (p, n_components)
        self.y_loadings_ = None     # Q: (q, n_components)
        self.x_scores_ = None       # T: (n, n_components)
        self.y_scores_ = None       # U: (n, n_components)
        self.x_mean_ = None
        self.x_std_ = None
        self.y_encoder_ = None
        self.classes_ = None
        self.vip_ = None
        self.feature_names_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        """Fit sPLS-DA model.
        
        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y : ndarray of class labels (strings or integers)
        feature_names : list of str, optional
        """
        self.feature_names_ = feature_names
        
        # Encode Y as dummy matrix
        self.y_encoder_ = LabelBinarizer()
        Y = self.y_encoder_.fit_transform(y).astype(float)
        if Y.shape[1] == 1:  # binary case
            Y = np.hstack([1 - Y, Y])
        self.classes_ = self.y_encoder_.classes_
        
        n, p = X.shape
        q = Y.shape[1]
        ncomp = min(self.n_components, n - 1, p)
        
        # Resolve keepX
        if self.keepX is None:
            keepX = [p] * ncomp  # no sparsity
        elif isinstance(self.keepX, int):
            keepX = [self.keepX] * ncomp
        else:
            keepX = list(self.keepX)
            while len(keepX) < ncomp:
                keepX.append(keepX[-1])
        
        # Center (don't scale — scaling is done in preprocessing)
        self.x_mean_ = X.mean(axis=0)
        Xk = X - self.x_mean_
        y_mean = Y.mean(axis=0)
        Yk = Y - y_mean
        
        # Storage
        T = np.zeros((n, ncomp))  # X scores
        U = np.zeros((n, ncomp))  # Y scores
        W = np.zeros((p, ncomp))  # X weights
        P = np.zeros((p, ncomp))  # X loadings
        Q = np.zeros((q, ncomp))  # Y loadings
        
        for h in range(ncomp):
            # Initialize u as first column of Yk
            u = Yk[:, 0].copy()
            
            for _ in range(self.max_iter):
                # X weight: w = X'u / u'u
                w = Xk.T @ u
                norm_w = np.linalg.norm(w)
                if norm_w > 0:
                    w = w / norm_w
                
                # Apply sparsity
                w = _soft_threshold(w, keepX[h])
                
                # X score: t = Xw
                t = Xk @ w
                
                # Y loading: q = Y't / t't
                tt = t @ t
                if tt > 0:
                    q_h = Yk.T @ t / tt
                else:
                    q_h = np.zeros(q)
                
                # Y score: u = Yq / q'q
                qq = q_h @ q_h
                if qq > 0:
                    u_new = Yk @ q_h / qq
                else:
                    u_new = u
                
                # Check convergence
                if np.linalg.norm(u_new - u) < self.tol:
                    u = u_new
                    break
                u = u_new
            
            # X loading: p = X't / t't
            tt = t @ t
            if tt > 0:
                p_h = Xk.T @ t / tt
            else:
                p_h = np.zeros(p)
            
            # Store
            T[:, h] = t
            U[:, h] = u
            W[:, h] = w
            P[:, h] = p_h
            Q[:, h] = q_h
            
            # Deflate X and Y
            Xk = Xk - np.outer(t, p_h)
            Yk = Yk - np.outer(t, q_h)
        
        self.x_scores_ = T
        self.y_scores_ = U
        self.x_weights_ = W
        self.x_loadings_ = P
        self.y_loadings_ = Q
        self.n_components_ = ncomp
        
        # Compute VIP scores
        self.vip_ = self._compute_vip(X)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new X data onto fitted components."""
        Xc = X - self.x_mean_
        # Use rotation matrix: R = W(P'W)^{-1}
        PtW = self.x_loadings_.T @ self.x_weights_
        try:
            R = self.x_weights_ @ np.linalg.inv(PtW)
        except np.linalg.LinAlgError:
            R = self.x_weights_ @ np.linalg.pinv(PtW)
        return Xc @ R
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        scores = self.transform(X)
        # Project scores to Y space via Q loadings, then pick argmax class
        Y_hat = scores @ self.y_loadings_.T
        class_idx = np.argmax(Y_hat, axis=1)
        return self.classes_[class_idx]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        return accuracy_score(y, self.predict(X))
    
    def _compute_vip(self, X: np.ndarray) -> np.ndarray:
        """Variable Importance in Projection (VIP) scores.
        
        VIP_j = sqrt(p * sum_h(SS_h * w_jh^2) / sum_h(SS_h))
        where SS_h = t_h' * t_h * q_h' * q_h (explained variance per component)
        """
        p = X.shape[1]
        ncomp = self.n_components_
        
        SS = np.zeros(ncomp)
        for h in range(ncomp):
            t_h = self.x_scores_[:, h]
            q_h = self.y_loadings_[:, h]
            SS[h] = (t_h @ t_h) * (q_h @ q_h)
        
        total_SS = SS.sum()
        if total_SS == 0:
            return np.ones(p)
        
        vip = np.zeros(p)
        for j in range(p):
            s = 0
            for h in range(ncomp):
                s += SS[h] * self.x_weights_[j, h] ** 2
            vip[j] = np.sqrt(p * s / total_SS)
        
        return vip
    
    def get_vip_df(self):
        """Return VIP scores as a sorted DataFrame."""
        names = self.feature_names_ or [f"f{i}" for i in range(len(self.vip_))]
        df = pd.DataFrame({"Feature": names, "VIP": self.vip_})
        df["Important"] = df["VIP"] >= 1.0
        return df.sort_values("VIP", ascending=False).reset_index(drop=True)
    
    def get_loadings_df(self):
        """Return loadings as a DataFrame."""
        names = self.feature_names_ or [f"f{i}" for i in range(len(self.x_loadings_))]
        cols = [f"comp{h+1}" for h in range(self.n_components_)]
        return pd.DataFrame(self.x_weights_, index=names, columns=cols)
    
    def get_scores_df(self, sample_names=None):
        """Return sample scores as a DataFrame."""
        cols = [f"comp{h+1}" for h in range(self.n_components_)]
        idx = sample_names or list(range(self.x_scores_.shape[0]))
        return pd.DataFrame(self.x_scores_, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_splsda(X, y, n_components=2, keepX=None, cv=None):
    """Leave-one-out or k-fold CV for sPLS-DA.
    
    Returns
    -------
    dict with 'accuracy', 'predictions', 'true_labels', 'per_fold'
    """
    if cv is None:
        cv = LeaveOneOut()
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    predictions = np.empty(len(y), dtype=object)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        model = SPLSDA(n_components=n_components, keepX=keepX)
        model.fit(X_train, y_train)
        predictions[test_idx] = model.predict(X_test)
    
    acc = accuracy_score(y, predictions)
    return {"accuracy": acc, "predictions": predictions, "true_labels": y}


# ---------------------------------------------------------------------------
# DIABLO (multi-block sPLS-DA)
# ---------------------------------------------------------------------------

class DIABLO:
    """Multi-block sPLS-DA (DIABLO).
    
    Finds latent components maximally correlated across omics blocks
    while discriminating between classes.
    
    Parameters
    ----------
    n_components : int
        Number of latent components.
    keepX : dict of {block_name: list of int}, optional
        Features to keep per block per component. If None, no sparsity.
    design : float or ndarray
        Off-diagonal value for design matrix (inter-block connection strength).
        0 = no connection, 1 = full connection. Default 0.1.
    max_iter : int
        Maximum NIPALS iterations.
    tol : float
        Convergence tolerance.
    """
    
    def __init__(self, n_components=2, keepX=None, design=0.1, max_iter=500, tol=1e-6):
        self.n_components = n_components
        self.keepX = keepX
        self.design = design
        self.max_iter = max_iter
        self.tol = tol
        
        # Fitted attributes
        self.block_weights_ = {}    # {block: W matrix}
        self.block_loadings_ = {}   # {block: P matrix}
        self.block_scores_ = {}     # {block: T matrix}
        self.y_loadings_ = None
        self.y_encoder_ = None
        self.classes_ = None
        self.block_means_ = {}
        self.block_vip_ = {}
        self.block_names_ = None
        self.feature_names_ = {}
        self.design_matrix_ = None
        self.correlations_ = None
    
    def fit(self, X_blocks: dict, y: np.ndarray, feature_names: dict = None):
        """Fit DIABLO model.
        
        Parameters
        ----------
        X_blocks : dict of {block_name: ndarray (n, p_k)}
        y : ndarray of class labels
        feature_names : dict of {block_name: list of str}
        """
        self.block_names_ = list(X_blocks.keys())
        self.feature_names_ = feature_names or {k: None for k in self.block_names_}
        K = len(self.block_names_)
        
        # Encode Y
        self.y_encoder_ = LabelBinarizer()
        Y = self.y_encoder_.fit_transform(y).astype(float)
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])
        self.classes_ = self.y_encoder_.classes_
        
        n = Y.shape[0]
        q = Y.shape[1]
        
        # Build design matrix
        if isinstance(self.design, (int, float)):
            self.design_matrix_ = np.full((K + 1, K + 1), self.design)
            np.fill_diagonal(self.design_matrix_, 0)
            # Last row/col is Y block — always connected
            self.design_matrix_[:, -1] = 1.0
            self.design_matrix_[-1, :] = 1.0
            self.design_matrix_[-1, -1] = 0.0
        else:
            self.design_matrix_ = np.array(self.design)
        
        # Resolve keepX
        if self.keepX is None:
            keepX = {name: [X_blocks[name].shape[1]] * self.n_components 
                     for name in self.block_names_}
        else:
            keepX = {}
            for name in self.block_names_:
                if name in self.keepX:
                    kx = self.keepX[name]
                    if isinstance(kx, int):
                        kx = [kx] * self.n_components
                    keepX[name] = list(kx)
                else:
                    keepX[name] = [X_blocks[name].shape[1]] * self.n_components
        
        ncomp = min(self.n_components, n - 1)
        
        # Center blocks
        Xk = {}
        for name in self.block_names_:
            self.block_means_[name] = X_blocks[name].mean(axis=0)
            Xk[name] = X_blocks[name] - self.block_means_[name]
        
        y_mean = Y.mean(axis=0)
        Yk = Y - y_mean
        
        # Initialize storage
        for name in self.block_names_:
            p_k = X_blocks[name].shape[1]
            self.block_weights_[name] = np.zeros((p_k, ncomp))
            self.block_loadings_[name] = np.zeros((p_k, ncomp))
            self.block_scores_[name] = np.zeros((n, ncomp))
        
        self.y_loadings_ = np.zeros((q, ncomp))
        y_scores = np.zeros((n, ncomp))
        
        for h in range(ncomp):
            # Initialize: super score as first SVD component of concatenated blocks
            combined = np.hstack([Xk[name] for name in self.block_names_])
            U, S, Vt = np.linalg.svd(combined, full_matrices=False)
            super_t = U[:, 0] * S[0]
            
            for iteration in range(self.max_iter):
                t_blocks = {}
                w_blocks = {}
                
                for ki, name in enumerate(self.block_names_):
                    # Compute weighted combination of other blocks' scores + Y
                    # a_k = sum_{j!=k} design[k,j] * X_j' * t_j + design[k,Y] * Y' * u
                    target = np.zeros(n)
                    
                    # Contribution from other X blocks
                    for kj, name_j in enumerate(self.block_names_):
                        if ki != kj and ki < len(self.design_matrix_) and kj < len(self.design_matrix_):
                            if name_j in t_blocks:
                                target += self.design_matrix_[ki, kj] * t_blocks[name_j]
                            else:
                                target += self.design_matrix_[ki, kj] * super_t
                    
                    # Contribution from Y block
                    target += self.design_matrix_[ki, -1] * Yk @ (Yk.T @ super_t) / max(super_t @ super_t, 1e-10)
                    
                    # Fallback: if target is zero, use super_t
                    if np.linalg.norm(target) < 1e-10:
                        target = super_t
                    
                    # X weight
                    w_k = Xk[name].T @ target
                    norm_w = np.linalg.norm(w_k)
                    if norm_w > 0:
                        w_k = w_k / norm_w
                    
                    # Sparsity
                    w_k = _soft_threshold(w_k, keepX[name][h])
                    w_blocks[name] = w_k
                    
                    # Block score
                    t_blocks[name] = Xk[name] @ w_k
                
                # Update super score as weighted average of block scores
                super_t_new = np.zeros(n)
                for ki, name in enumerate(self.block_names_):
                    super_t_new += t_blocks[name]
                super_t_new /= K
                
                # Check convergence
                if np.linalg.norm(super_t_new - super_t) < self.tol * np.linalg.norm(super_t + 1e-10):
                    super_t = super_t_new
                    break
                super_t = super_t_new
            
            # Store weights, scores, loadings
            for name in self.block_names_:
                self.block_weights_[name][:, h] = w_blocks[name]
                t_k = t_blocks[name]
                self.block_scores_[name][:, h] = t_k
                tt = t_k @ t_k
                if tt > 0:
                    self.block_loadings_[name][:, h] = Xk[name].T @ t_k / tt
            
            # Y loading
            st = super_t @ super_t
            if st > 0:
                self.y_loadings_[:, h] = Yk.T @ super_t / st
            y_scores[:, h] = super_t
            
            # Deflate all blocks and Y using super score
            for name in self.block_names_:
                t_k = self.block_scores_[name][:, h]
                tt = t_k @ t_k
                if tt > 0:
                    p_k = Xk[name].T @ t_k / tt
                    Xk[name] = Xk[name] - np.outer(t_k, p_k)
            
            st = super_t @ super_t
            if st > 0:
                Yk = Yk - np.outer(super_t, self.y_loadings_[:, h])
        
        self.n_components_ = ncomp
        
        # Compute per-block VIP
        for name in self.block_names_:
            self.block_vip_[name] = self._compute_block_vip(name, X_blocks[name])
        
        # Compute inter-block correlations
        self.correlations_ = self._compute_correlations()
        
        return self
    
    def _compute_block_vip(self, block_name: str, X: np.ndarray) -> np.ndarray:
        """VIP scores for a single block within DIABLO."""
        p = X.shape[1]
        ncomp = self.n_components_
        W = self.block_weights_[block_name]
        T = self.block_scores_[block_name]
        Q = self.y_loadings_
        
        SS = np.zeros(ncomp)
        for h in range(ncomp):
            SS[h] = (T[:, h] @ T[:, h]) * (Q[:, h] @ Q[:, h])
        
        total_SS = SS.sum()
        if total_SS == 0:
            return np.ones(p)
        
        vip = np.zeros(p)
        for j in range(p):
            s = sum(SS[h] * W[j, h] ** 2 for h in range(ncomp))
            vip[j] = np.sqrt(p * s / total_SS)
        
        return vip
    
    def _compute_correlations(self) -> pd.DataFrame:
        """Pairwise correlations between block score vectors (comp 1)."""
        names = self.block_names_
        K = len(names)
        corr = np.eye(K)
        for i in range(K):
            for j in range(i + 1, K):
                r = np.corrcoef(self.block_scores_[names[i]][:, 0],
                                self.block_scores_[names[j]][:, 0])[0, 1]
                corr[i, j] = r
                corr[j, i] = r
        return pd.DataFrame(corr, index=names, columns=names)
    
    def predict(self, X_blocks: dict) -> np.ndarray:
        """Predict class labels from multiple blocks."""
        # Average scores across blocks, project to Y space
        n = list(X_blocks.values())[0].shape[0]
        avg_scores = np.zeros((n, self.n_components_))
        
        for name in self.block_names_:
            Xc = X_blocks[name] - self.block_means_[name]
            W = self.block_weights_[name]
            P = self.block_loadings_[name]
            PtW = P.T @ W
            try:
                R = W @ np.linalg.inv(PtW)
            except np.linalg.LinAlgError:
                R = W @ np.linalg.pinv(PtW)
            avg_scores += Xc @ R
        
        avg_scores /= len(self.block_names_)
        Y_hat = avg_scores @ self.y_loadings_.T
        class_idx = np.argmax(Y_hat, axis=1)
        return self.classes_[class_idx]
    
    def score(self, X_blocks: dict, y: np.ndarray) -> float:
        """Classification accuracy."""
        return accuracy_score(y, self.predict(X_blocks))
    
    def get_vip_df(self, block_name: str):
        """VIP scores for a block as sorted DataFrame."""
        vip = self.block_vip_[block_name]
        names = self.feature_names_.get(block_name) or [f"f{i}" for i in range(len(vip))]
        df = pd.DataFrame({"Feature": names, "VIP": vip, "Block": block_name})
        df["Important"] = df["VIP"] >= 1.0
        return df.sort_values("VIP", ascending=False).reset_index(drop=True)
    
    def get_all_vip_df(self):
        """Combined VIP DataFrame across all blocks."""
        dfs = [self.get_vip_df(name) for name in self.block_names_]
        return pd.concat(dfs, ignore_index=True)
    
    def get_selected_features(self, block_name: str, component: int = 0):
        """Features with non-zero weights for a given component."""
        W = self.block_weights_[block_name][:, component]
        names = self.feature_names_.get(block_name) or [f"f{i}" for i in range(len(W))]
        mask = W != 0
        selected = [(names[i], W[i]) for i in range(len(W)) if mask[i]]
        if not selected:
            return pd.DataFrame(columns=["Feature", "Loading"])
        df = pd.DataFrame(selected, columns=["Feature", "Loading"])
        return df.sort_values("Loading", key=abs, ascending=False).reset_index(drop=True)


def cross_validate_diablo(X_blocks, y, n_components=2, keepX=None, design=0.1, cv=None):
    """LOO or k-fold CV for DIABLO.
    
    Returns
    -------
    dict with 'accuracy', 'predictions', 'true_labels'
    """
    if cv is None:
        cv = LeaveOneOut()
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    n = len(y)
    predictions = np.empty(n, dtype=object)
    indices = np.arange(n)
    
    for train_idx, test_idx in cv.split(indices, y):
        X_train = {name: X_blocks[name][train_idx] for name in X_blocks}
        X_test = {name: X_blocks[name][test_idx] for name in X_blocks}
        y_train = y[train_idx]
        
        model = DIABLO(n_components=n_components, keepX=keepX, design=design)
        model.fit(X_train, y_train)
        predictions[test_idx] = model.predict(X_test)
    
    acc = accuracy_score(y, predictions)
    return {"accuracy": acc, "predictions": predictions, "true_labels": y}


# ---------------------------------------------------------------------------
# Permutation testing
# ---------------------------------------------------------------------------

def _early_stop_check(null_accs_so_far, true_acc, n_done, n_max, alpha=0.05, confidence=0.99):
    """Check whether we can stop permutation testing early.
    
    Uses a sequential decision rule: after n_done permutations, compute a 
    confidence interval for the true p-value using the Clopper-Pearson 
    (exact binomial) interval. If the entire interval is above or below alpha,
    the conclusion won't change with more permutations.
    
    Parameters
    ----------
    null_accs_so_far : ndarray of null accuracies collected so far
    true_acc : float — observed accuracy
    n_done : int — permutations completed
    alpha : float — significance threshold (default 0.05)
    confidence : float — confidence level for the decision (default 0.99)
    
    Returns
    -------
    bool — True if we can stop early
    """
    from scipy.stats import beta as beta_dist
    
    k = np.sum(null_accs_so_far[:n_done] >= true_acc)  # successes
    
    # Clopper-Pearson interval for p-value
    tail = (1 - confidence) / 2
    if k == 0:
        p_lower = 0.0
    else:
        p_lower = beta_dist.ppf(tail, k, n_done - k + 1)
    
    if k == n_done:
        p_upper = 1.0
    else:
        p_upper = beta_dist.ppf(1 - tail, k + 1, n_done - k)
    
    # Can stop if entire CI is on one side of alpha
    if p_lower > alpha:  # clearly non-significant
        return True
    if p_upper < alpha:  # clearly significant
        return True
    return False


def permutation_test_splsda(X, y, n_components=2, keepX=None, n_permutations=1000,
                             random_state=42, early_stop=True, min_perms=50, check_every=25):
    """Permutation test for sPLS-DA: is LOO accuracy better than chance?
    
    Shuffles class labels n_permutations times, computes LOO accuracy each time
    to build a null distribution. With early_stop=True, checks periodically and
    stops when the significance conclusion is already clear.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray of class labels
    n_components : int
    keepX : int or list, optional
    n_permutations : int — maximum permutations
    random_state : int
    early_stop : bool — enable dynamic stopping (default True)
    min_perms : int — minimum permutations before early stopping kicks in
    check_every : int — check stopping criterion every N permutations
    
    Returns
    -------
    dict with 'true_accuracy', 'null_distribution', 'p_value', 'mean_null',
              'std_null', 'n_permutations_run', 'stopped_early'
    """
    rng = np.random.RandomState(random_state)
    
    true_result = cross_validate_splsda(X, y, n_components=n_components, keepX=keepX)
    true_acc = true_result["accuracy"]
    
    null_accs = np.zeros(n_permutations)
    n_run = 0
    stopped_early = False
    
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_result = cross_validate_splsda(X, y_perm, n_components=n_components, keepX=keepX)
        null_accs[i] = perm_result["accuracy"]
        n_run = i + 1
        
        # Early stopping check
        if early_stop and n_run >= min_perms and n_run % check_every == 0:
            if _early_stop_check(null_accs, true_acc, n_run, n_permutations):
                stopped_early = True
                break
    
    null_accs = null_accs[:n_run]
    p_value = (np.sum(null_accs >= true_acc) + 1) / (n_run + 1)
    
    return {
        "true_accuracy": true_acc,
        "null_distribution": null_accs,
        "p_value": p_value,
        "mean_null": null_accs.mean(),
        "std_null": null_accs.std(),
        "n_permutations_run": n_run,
        "stopped_early": stopped_early,
    }


def permutation_test_diablo(X_blocks, y, n_components=2, keepX=None, design=0.1,
                             n_permutations=1000, random_state=42,
                             early_stop=True, min_perms=50, check_every=25):
    """Permutation test for DIABLO with optional early stopping."""
    rng = np.random.RandomState(random_state)
    
    true_result = cross_validate_diablo(X_blocks, y, n_components=n_components,
                                         keepX=keepX, design=design)
    true_acc = true_result["accuracy"]
    
    null_accs = np.zeros(n_permutations)
    n_run = 0
    stopped_early = False
    
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_result = cross_validate_diablo(X_blocks, y_perm, n_components=n_components,
                                             keepX=keepX, design=design)
        null_accs[i] = perm_result["accuracy"]
        n_run = i + 1
        
        if early_stop and n_run >= min_perms and n_run % check_every == 0:
            if _early_stop_check(null_accs, true_acc, n_run, n_permutations):
                stopped_early = True
                break
    
    null_accs = null_accs[:n_run]
    p_value = (np.sum(null_accs >= true_acc) + 1) / (n_run + 1)
    
    return {
        "true_accuracy": true_acc,
        "null_distribution": null_accs,
        "p_value": p_value,
        "mean_null": null_accs.mean(),
        "std_null": null_accs.std(),
        "n_permutations_run": n_run,
        "stopped_early": stopped_early,
    }


# ---------------------------------------------------------------------------
# Stability selection (bootstrap)
# ---------------------------------------------------------------------------

def stability_selection_splsda(X, y, feature_names=None, n_components=2, keepX=None,
                                n_bootstrap=100, random_state=42):
    """Bootstrap stability selection for sPLS-DA feature selection.
    
    Fits sparse PLS-DA on n_bootstrap resampled datasets and records which
    features receive non-zero sparse weights in each run.  Features selected
    in a high fraction of bootstraps are genuinely stable.
    
    When *keepX* is ``None`` (no sparsity), falls back to VIP >= 1 as the
    selection criterion, but **sparse keepX should always be provided** for
    meaningful stability results.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray of class labels
    feature_names : list of str
    n_components : int
    keepX : int or list, optional
        Number of features to retain per component.  **Should be provided**
        so that the model is truly sparse.
    n_bootstrap : int
    random_state : int
    
    Returns
    -------
    DataFrame with Feature, Selection_Frequency, Mean_VIP, Std_VIP, Stable (freq >= 0.8)
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    p = X.shape[1]
    names = feature_names or [f"f{i}" for i in range(p)]
    
    # Determine whether the model is truly sparse
    _sparse = keepX is not None
    
    selection_counts = np.zeros(p)
    vip_accumulator = np.zeros((n_bootstrap, p))
    
    for b in range(n_bootstrap):
        # Stratified bootstrap: resample within each class
        idx = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            boot_idx = rng.choice(cls_idx, size=len(cls_idx), replace=True)
            idx.extend(boot_idx)
        idx = np.array(idx)
        
        X_boot = X[idx]
        y_boot = y[idx]
        
        model = SPLSDA(n_components=n_components, keepX=keepX)
        model.fit(X_boot, y_boot, feature_names=names)
        
        vip = model.vip_
        vip_accumulator[b, :] = vip
        
        if _sparse:
            # True sparse criterion: feature has non-zero weight on any component
            selected = np.any(model.x_weights_ != 0, axis=1)
            selection_counts[selected] += 1
        else:
            # Fallback for non-sparse models
            selection_counts[vip >= 1.0] += 1
    
    freq = selection_counts / n_bootstrap
    mean_vip = vip_accumulator.mean(axis=0)
    std_vip = vip_accumulator.std(axis=0)
    
    df = pd.DataFrame({
        "Feature": names,
        "Selection_Frequency": freq,
        "Mean_VIP": mean_vip,
        "Std_VIP": std_vip,
        "Stable": freq >= 0.8,
    }).sort_values("Selection_Frequency", ascending=False).reset_index(drop=True)
    
    return df


def stability_selection_diablo(X_blocks, y, feature_names=None, n_components=2,
                                keepX=None, design=0.1, n_bootstrap=100,
                                random_state=42):
    """Bootstrap stability selection for DIABLO feature selection.

    Fits the multi-block sPLS-DA (DIABLO) model on *n_bootstrap* stratified
    bootstrap resamples and records, for each omics block, which features
    receive non-zero sparse weights in each run.

    When *keepX* is ``None`` (no sparsity), falls back to VIP >= 1 as the
    selection criterion.

    Parameters
    ----------
    X_blocks : dict of {block_name: ndarray (n_samples, p_k)}
    y : ndarray of class labels
    feature_names : dict of {block_name: list of str}, optional
    n_components : int
    keepX : dict of {block_name: list of int}, optional
    design : float
    n_bootstrap : int
    random_state : int

    Returns
    -------
    dict of {block_name: DataFrame} with columns
        Feature, Selection_Frequency, Mean_VIP, Std_VIP, Stable.
    """
    rng = np.random.RandomState(random_state)
    block_names = list(X_blocks.keys())
    n = len(y)
    
    # Determine whether model is truly sparse
    _sparse = keepX is not None

    # Resolve feature names
    if feature_names is None:
        feature_names = {name: [f"{name}_f{i}" for i in range(X_blocks[name].shape[1])]
                         for name in block_names}

    # Accumulators per block
    counts = {name: np.zeros(X_blocks[name].shape[1]) for name in block_names}
    vips = {name: np.zeros((n_bootstrap, X_blocks[name].shape[1])) for name in block_names}

    for b in range(n_bootstrap):
        # Stratified bootstrap
        idx = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            boot_idx = rng.choice(cls_idx, size=len(cls_idx), replace=True)
            idx.extend(boot_idx)
        idx = np.array(idx)

        X_boot = {name: X_blocks[name][idx] for name in block_names}
        y_boot = y[idx]

        model = DIABLO(n_components=n_components, keepX=keepX, design=design)
        model.fit(X_boot, y_boot, feature_names=feature_names)

        for name in block_names:
            vip = model.block_vip_[name]
            vips[name][b, :] = vip
            if _sparse:
                # True sparse criterion: non-zero weight on any component
                weights = model.block_weights_[name]
                selected = np.any(weights != 0, axis=1)
                counts[name][selected] += 1
            else:
                counts[name][vip >= 1.0] += 1

    results = {}
    for name in block_names:
        freq = counts[name] / n_bootstrap
        mean_vip = vips[name].mean(axis=0)
        std_vip = vips[name].std(axis=0)
        results[name] = pd.DataFrame({
            "Feature": feature_names[name],
            "Selection_Frequency": freq,
            "Mean_VIP": mean_vip,
            "Std_VIP": std_vip,
            "Stable": freq >= 0.8,
        }).sort_values("Selection_Frequency", ascending=False).reset_index(drop=True)

    return results