import os, json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from utils import db
os.makedirs("models", exist_ok=True)

def _prep(df: pd.DataFrame):
    y = df["repay_good"].values
    X = df.drop(columns=["repay_good"])
    # tolerate missing demographic features by filling
    for col in ["gender","age","dependents","occupation","income","branch","product","loan_amount","status","loan_status","is_fraud","application_date","customer_name","id_reg_number","ref_number","loan_type","loan_health","created_date"]:
        if col not in X.columns:
            # Create safe defaults
            if col in ("age","dependents","income","loan_amount","is_fraud"):
                X[col] = 0
            else:
                X[col] = "unknown"
    # drop texty/non-predictive columns
    X = X.drop(columns=["customer_name","application_date","id_reg_number","ref_number","created_date"], errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="passthrough")
    return X, y, pre

def _fit_family(family: str, pre, X_train, y_train):
    if family=="LogReg":
        clf = LogisticRegression(max_iter=1000)
    elif family=="RF":
        clf = RandomForestClassifier(n_estimators=250, max_depth=None, n_jobs=-1)
    elif family=="GB":
        clf = GradientBoostingClassifier()
    elif family=="XGBoost":
        if xgb is None:
            return None, "xgboost not installed"
        clf = xgb.XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
    elif family=="LightGBM":
        if lgb is None:
            return None, "lightgbm not installed"
        clf = lgb.LGBMClassifier(n_estimators=350, max_depth=-1, num_leaves=31, learning_rate=0.08)
    elif family=="Hybrid":
        # Simple hybrid: average predicted probabilities of RF + GB (stack-lite)
        rf = RandomForestClassifier(n_estimators=220, n_jobs=-1)
        gb = GradientBoostingClassifier()
        from sklearn.base import BaseEstimator, ClassifierMixin
        class Hybrid(BaseEstimator, ClassifierMixin):
            def __init__(self, rf, gb):
                self.rf = rf; self.gb = gb
            def fit(self, X, y):
                self.rf.fit(X, y); self.gb.fit(X, y); return self
            def predict_proba(self, X):
                p1 = self.rf.predict_proba(X)[:,1]
                p2 = self.gb.predict_proba(X)[:,1]
                p = (p1+p2)/2
                return np.vstack([1-p, p]).T
            def predict(self, X):
                return (self.predict_proba(X)[:,1] >= 0.5).astype(int)
        clf = Hybrid(rf, gb)
    else:
        raise ValueError("Unknown family")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe, None

def train_and_version(df: pd.DataFrame, families=None, test_size=0.2, seed=42):
    if families is None:
        families = ["LogReg","RF","GB","XGBoost","LightGBM","Hybrid"]
    X, y, pre = _prep(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    results = []
    for fam in families:
        model, err = _fit_family(fam, pre, X_train, y_train)
        if model is None:
            metrics = {"status":"skipped","reason": err}
            version = 1 + sum(1 for r in db.list_models() if r[0]==fam)
            path = f"models/{fam}_v{version}.joblib"
            db.insert_model(fam, version, metrics, path, deployed=0)
            results.append((fam, version, metrics, path))
            continue

        proba = model.predict_proba(X_test)[:,1]
        auc = float(roc_auc_score(y_test, proba))
        acc = float(accuracy_score(y_test, (proba>=0.5).astype(int)))
        rec = float(recall_score(y_test, (proba>=0.5).astype(int)))
        metrics = {"AUC": round(auc,4), "accuracy": round(acc,4), "recall": round(rec,4)}
        version = 1 + sum(1 for r in db.list_models() if r[0]==fam)
        path = f"models/{fam}_v{version}.joblib"
        joblib.dump(model, path)
        db.insert_model(fam, version, metrics, path, deployed=0)
        results.append((fam, version, metrics, path))
    # Auto-deploy best by AUC among trained ones
    trained = [r for r in results if "AUC" in r[2]]
    if trained:
        best = max(trained, key=lambda r: r[2]["AUC"])
        fam, ver, _, _ = best
        db.mark_deployed(fam, ver)
    return results

def load_deployed():
    rows = db.list_models()
    for fam, ver, metrics, path, deployed, created in rows:
        if deployed==1 and os.path.exists(path):
            return fam, ver, json.loads(metrics), path
    return None