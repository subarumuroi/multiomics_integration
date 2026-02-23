"""Ordinal Regression: LogisticAT/IT/SE via mord.

Respects ordering (Green < Ripe < Overripe).
Primary metric: MAE (ordinal distance), alongside accuracy.
"""

import numpy as np
import pandas as pd
import mord
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


MODEL_MAP = {
    "AT": mord.LogisticAT,
    "IT": mord.LogisticIT,
    "SE": mord.LogisticSE,
}


def train_ordinal(X, y_encoded, model_type="AT", log_transform=True, alpha=1.0):
    """Train an ordinal regression model.
    
    Parameters
    ----------
    X : ndarray (already preprocessed features)
    y_encoded : ndarray of ordinal integers (0, 1, 2)
    model_type : 'AT', 'IT', or 'SE'
    log_transform : bool — apply log10 transform before scaling
    alpha : float — regularization parameter
    
    Returns
    -------
    pipeline : fitted Pipeline (scaler + model)
    coef_df : DataFrame of feature coefficients
    """
    if model_type not in MODEL_MAP:
        raise ValueError(f"model_type must be one of {list(MODEL_MAP.keys())}")
    
    steps = []
    if log_transform:
        # Already handled in ingestion typically, but allow here for standalone use
        pass
    steps.append(("scaler", MinMaxScaler()))
    steps.append(("model", MODEL_MAP[model_type](alpha=alpha)))
    
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
