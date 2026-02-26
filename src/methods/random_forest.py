"""Random Forest classifier with feature importance, SHAP, cross-validation, and permutation testing."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance as sklearn_perm_importance


def train_rf(X, y, feature_names=None, n_estimators=100, max_depth=3, random_state=42):
    """Train a Random Forest classifier.
    
    Returns
    -------
    model : fitted RandomForestClassifier
    importance_df : DataFrame of feature importances (Gini)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X, y)
    
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    importance_df = pd.DataFrame({
        "Feature": names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    
    return model, importance_df


def cross_validate_rf(X, y, n_estimators=100, max_depth=3, random_state=42, cv=None):
    """LOO or k-fold CV for Random Forest.
    
    Returns
    -------
    dict with 'accuracy', 'predictions', 'true_labels', 'confusion_matrix'
    """
    if cv is None:
        cv = LeaveOneOut()
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    
    predictions = cross_val_predict(model, X, y, cv=cv)
    acc = accuracy_score(y, predictions)
    cm = confusion_matrix(y, predictions)
    
    return {
        "accuracy": acc,
        "predictions": predictions,
        "true_labels": y,
        "confusion_matrix": cm,
        "classification_report": classification_report(y, predictions, output_dict=True),
    }


def compute_permutation_importance(model, X, y, feature_names=None, n_repeats=10, random_state=42):
    """Permutation importance (model-agnostic).
    
    Returns DataFrame with mean, std of importance per feature.
    """
    result = sklearn_perm_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame({
        "Feature": names,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)
    return df


def compute_shap_values(model, X, feature_names=None):
    """SHAP values using TreeExplainer.
    
    Returns
    -------
    shap_values : SHAP explanation object
    shap_importance_df : DataFrame of mean |SHAP| per feature
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    
    # Mean absolute SHAP across all classes and samples
    if isinstance(shap_values.values, list):
        # Multi-class: list of arrays
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values.values], axis=0)
    else:
        vals = shap_values.values
        if vals.ndim == 3:
            mean_abs = np.abs(vals).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(vals).mean(axis=0)
    
    df = pd.DataFrame({
        "Feature": names,
        "Mean_Abs_SHAP": mean_abs,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    return shap_values, df


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _early_stop_check_rf(null_accs_so_far, true_acc, n_done, alpha=0.05, confidence=0.99):
    """Sequential stopping rule using Clopper-Pearson CI for the p-value."""
    from scipy.stats import beta as beta_dist

    k = int(np.sum(null_accs_so_far[:n_done] >= true_acc))
    tail = (1 - confidence) / 2

    p_lower = 0.0 if k == 0 else beta_dist.ppf(tail, k, n_done - k + 1)
    p_upper = 1.0 if k == n_done else beta_dist.ppf(1 - tail, k + 1, n_done - k)

    return p_lower > alpha or p_upper < alpha


def permutation_test_rf(X, y, n_estimators=100, max_depth=3, random_state=42,
                         n_permutations=200, early_stop=True, min_perms=50,
                         check_every=25):
    """Permutation test for Random Forest: is LOO accuracy better than chance?

    Shuffles class labels *n_permutations* times and computes LOO accuracy for
    each shuffle to build a null distribution.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray of class labels
    n_estimators, max_depth, random_state : RF hyperparameters
    n_permutations : int — maximum permutations
    early_stop : bool — enable dynamic stopping (default True)
    min_perms : int — minimum permutations before early stopping kicks in
    check_every : int — check stopping criterion every N permutations

    Returns
    -------
    dict with 'true_accuracy', 'null_distribution', 'p_value', 'mean_null',
              'std_null', 'n_permutations_run', 'stopped_early'
    """
    rng = np.random.RandomState(random_state)

    true_result = cross_validate_rf(X, y, n_estimators=n_estimators,
                                     max_depth=max_depth, random_state=random_state)
    true_acc = true_result["accuracy"]

    null_accs = np.zeros(n_permutations)
    n_run = 0
    stopped_early = False

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_result = cross_validate_rf(X, y_perm, n_estimators=n_estimators,
                                         max_depth=max_depth, random_state=random_state)
        null_accs[i] = perm_result["accuracy"]
        n_run = i + 1

        if early_stop and n_run >= min_perms and n_run % check_every == 0:
            if _early_stop_check_rf(null_accs, true_acc, n_run):
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
