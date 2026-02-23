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
        import pandas as pd
        names = self.feature_names_ or [f"f{i}" for i in range(len(self.vip_))]
        df = pd.DataFrame({"Feature": names, "VIP": self.vip_})
        df["Important"] = df["VIP"] >= 1.0
        return df.sort_values("VIP", ascending=False).reset_index(drop=True)
    
    def get_loadings_df(self):
        """Return loadings as a DataFrame."""
        import pandas as pd
        names = self.feature_names_ or [f"f{i}" for i in range(len(self.x_loadings_))]
        cols = [f"comp{h+1}" for h in range(self.n_components_)]
        return pd.DataFrame(self.x_weights_, index=names, columns=cols)
    
    def get_scores_df(self, sample_names=None):
        """Return sample scores as a DataFrame."""
        import pandas as pd
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
    
    def _compute_correlations(self) -> np.ndarray:
        """Pairwise correlations between block score vectors (comp 1)."""
        import pandas as pd
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
        import pandas as pd
        vip = self.block_vip_[block_name]
        names = self.feature_names_.get(block_name) or [f"f{i}" for i in range(len(vip))]
        df = pd.DataFrame({"Feature": names, "VIP": vip, "Block": block_name})
        n = len(vip)
        # Percentile ranking within block
        ranks = pd.Series(vip).rank(pct=True) * 100
        df["Percentile"] = ranks.values
        df["Important"] = df["Percentile"] >= 50
        return df.sort_values("VIP", ascending=False).reset_index(drop=True)
    
    def get_all_vip_df(self):
        """Combined VIP DataFrame across all blocks."""
        import pandas as pd
        dfs = [self.get_vip_df(name) for name in self.block_names_]
        return pd.concat(dfs, ignore_index=True)
    
    def get_selected_features(self, block_name: str, component: int = 0):
        """Features with non-zero weights for a given component."""
        import pandas as pd
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
