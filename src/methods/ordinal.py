"""Ordinal Regression: LogisticAT/IT/SE via mord.

Respects ordering (Green < Ripe < Overripe).
Primary metric: MAE (ordinal distance), alongside accuracy.
"""

import numpy as np
import pandas as pd
import mord
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.pipeline import Pipeline


MODEL_MAP = {
    "AT": mord.LogisticAT,
    "IT": mord.LogisticIT,
    "SE": mord.LogisticSE,
}


def train_ordinal(X, y_encoded, model_type="AT", alpha=1.0):
    """Train an ordinal regression model.
    
    Parameters
    ----------
    X : ndarray (already preprocessed features)
    y_encoded : ndarray of ordinal integers (0, 1, 2)
    model_type : 'AT', 'IT', or 'SE'
    alpha : float — regularization parameter
    
    Returns
    -------
    pipeline : fitted Pipeline (MinMaxScaler + mord model)
    coef : ndarray of model coefficients
    """
    if model_type not in MODEL_MAP:
        raise ValueError(f"model_type must be one of {list(MODEL_MAP.keys())}")
    
    steps = [
        ("scaler", MinMaxScaler()),
        ("model", MODEL_MAP[model_type](alpha=alpha)),
    ]
    
    pipeline = Pipeline(steps)
    pipeline.fit(X, y_encoded)
    
    model = pipeline.named_steps["model"]
    coef = model.coef_.flatten() if hasattr(model, "coef_") else np.zeros(X.shape[1])
    
    return pipeline, coef


def get_coefficient_df(coef, feature_names=None):
    """Return ordinal regression coefficients as a sorted DataFrame."""
    names = feature_names or [f"f{i}" for i in range(len(coef))]
    df = pd.DataFrame({
        "Feature": names,
        "Coefficient": coef,
        "Abs_Coefficient": np.abs(coef),
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    return df


def cross_validate_ordinal(X, y_encoded, model_type="AT", alpha=1.0, cv=None):
    """LOO or k-fold CV for ordinal regression.
    
    Returns
    -------
    dict with 'accuracy', 'mae', 'predictions', 'true_labels', 
    'confusion_matrix', 'mean_coefficients'
    """
    if cv is None:
        cv = LeaveOneOut()
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    n = len(y_encoded)
    predictions = np.zeros(n, dtype=int)
    all_coefs = []
    
    for train_idx, test_idx in cv.split(X, y_encoded):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_encoded[train_idx]
        
        pipeline, coef = train_ordinal(X_train, y_train, model_type=model_type, alpha=alpha)
        predictions[test_idx] = pipeline.predict(X_test).astype(int)
        all_coefs.append(coef)
    
    acc = accuracy_score(y_encoded, predictions)
    mae = mean_absolute_error(y_encoded, predictions)
    cm = confusion_matrix(y_encoded, predictions)
    mean_coef = np.mean(all_coefs, axis=0)
    
    return {
        "accuracy": acc,
        "mae": mae,
        "predictions": predictions,
        "true_labels": y_encoded,
        "confusion_matrix": cm,
        "mean_coefficients": mean_coef,
    }


def compare_ordinal_models(X, y_encoded, feature_names=None, cv=None):
    """Compare AT, IT, SE side by side.
    
    Returns DataFrame with accuracy and MAE for each model type.
    """
    results = []
    for model_type in ["AT", "IT", "SE"]:
        cv_result = cross_validate_ordinal(X, y_encoded, model_type=model_type, cv=cv)
        results.append({
            "Model": f"Logistic{model_type}",
            "Accuracy": cv_result["accuracy"],
            "MAE": cv_result["mae"],
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _early_stop_check_ord(null_accs_so_far, true_acc, n_done, alpha=0.05, confidence=0.99):
    """Sequential stopping rule using Clopper-Pearson CI for the p-value."""
    from scipy.stats import beta as beta_dist

    k = int(np.sum(null_accs_so_far[:n_done] >= true_acc))
    tail = (1 - confidence) / 2

    p_lower = 0.0 if k == 0 else beta_dist.ppf(tail, k, n_done - k + 1)
    p_upper = 1.0 if k == n_done else beta_dist.ppf(1 - tail, k + 1, n_done - k)

    return p_lower > alpha or p_upper < alpha


def permutation_test_ordinal(X, y_encoded, model_type="AT", alpha=1.0,
                              n_permutations=200, random_state=42,
                              early_stop=True, min_perms=50, check_every=25):
    """Permutation test for ordinal regression: is LOO accuracy better than chance?

    Shuffles ordinal labels *n_permutations* times and computes LOO accuracy
    for each shuffle to build a null distribution.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y_encoded : ndarray of ordinal integers (0, 1, 2, ...)
    model_type : str — 'AT', 'IT', or 'SE'
    alpha : float — regularization parameter
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

    true_result = cross_validate_ordinal(X, y_encoded, model_type=model_type, alpha=alpha)
    true_acc = true_result["accuracy"]

    null_accs = np.zeros(n_permutations)
    n_run = 0
    stopped_early = False

    for i in range(n_permutations):
        y_perm = rng.permutation(y_encoded)
        perm_result = cross_validate_ordinal(X, y_perm, model_type=model_type, alpha=alpha)
        null_accs[i] = perm_result["accuracy"]
        n_run = i + 1

        if early_stop and n_run >= min_perms and n_run % check_every == 0:
            if _early_stop_check_ord(null_accs, true_acc, n_run):
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
