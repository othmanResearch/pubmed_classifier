"""
Core training logic: stratified train/validation/test split, cross-validation,
held-out evaluation, and MLflow tracking.

"""

import json
import logging
import os
import pickle
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

DEFAULTS = {
    "test_size": 0.2,
    "val_size": 0.2,
    "random_state": 56,
    "n_splits": 10,
    "tfidf_min_df": 2,
    "tfidf_ngram_max": 3,
    "clf_C": 10,
    "mlflow_experiment": "pubmed_classifier",
}


def _cfg(config, key):
    return config.get(key, DEFAULTS.get(key))


def _read_pkl(path):
    with open(os.path.expanduser(path), "rb") as f:
        return pickle.load(f)


def load_dataset(config):
    """Load positive/negative [texts, pmids] pkls and build labelled arrays."""
    pos_texts, pos_pmids = _read_pkl(config["cls1"])
    neg_texts, neg_pmids = _read_pkl(config["cls2"])
    logger.info("positive: %d  negative: %d", len(pos_texts), len(neg_texts))
    texts = list(pos_texts) + list(neg_texts)
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    pmids = list(pos_pmids) + list(neg_pmids)
    return texts, labels, pmids


def split_dataset(texts, labels, test_size, val_size, random_state):
    """Stratified three-way split returning data plus original indices."""
    labels = np.asarray(labels)
    indices = np.arange(len(texts))
    X_pool, X_test, y_pool, y_test, idx_pool, idx_test = train_test_split(
        texts, labels, indices,
        test_size=test_size, random_state=random_state, stratify=labels,
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_pool, y_pool, idx_pool,
        test_size=val_size, random_state=random_state, stratify=y_pool,
    )
    return {
        "X_train": X_train, "y_train": y_train, "idx_train": idx_train,
        "X_val": X_val, "y_val": y_val, "idx_val": idx_val,
        "X_test": X_test, "y_test": y_test, "idx_test": idx_test,
    }


def build_pipeline(config):
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=_cfg(config, "tfidf_min_df"),
                                  ngram_range=(1, _cfg(config, "tfidf_ngram_max")))),
        ("clf", LogisticRegression(C=_cfg(config, "clf_C"),
                                   solver="liblinear",
                                   random_state=_cfg(config, "random_state"))),
    ])


def evaluate(pipeline, X, y):
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y, proba),
    }, pred, proba


def run_training(config):
    """Run the full train/validate/evaluate pipeline with MLflow tracking.

    Returns a dict of metrics and output paths.
    """
    random_state = _cfg(config, "random_state")
    n_splits = _cfg(config, "n_splits")

    texts, labels, pmids = load_dataset(config)
    split = split_dataset(texts, labels, _cfg(config, "test_size"),
                          _cfg(config, "val_size"), random_state)
    logger.info("split -> train %d | val %d | test %d",
                len(split["X_train"]), len(split["X_val"]), len(split["X_test"]))

    pipeline = build_pipeline(config)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, split["X_train"], split["y_train"],
                                cv=cv, scoring="accuracy")

    pipeline.fit(split["X_train"], split["y_train"])
    val_metrics, _, _ = evaluate(pipeline, split["X_val"], split["y_val"])
    test_metrics, y_pred, y_proba = evaluate(pipeline, split["X_test"], split["y_test"])

    cv_metrics = {
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
    }
    logger.info("CV %s | VAL %s | TEST %s", cv_metrics, val_metrics, test_metrics)

    _track_mlflow(config, pipeline, cv_scores, cv_metrics, val_metrics,
                  test_metrics, split, y_pred, y_proba)
    outputs = _save_artifacts(config, pipeline, split, pmids, labels,
                              cv_metrics, val_metrics, test_metrics, y_pred, y_proba)

    return {"cv": cv_metrics, "validation": val_metrics, "test": test_metrics, **outputs}


def _track_mlflow(config, pipeline, cv_scores, cv_metrics, val_metrics,
                  test_metrics, split, y_pred, y_proba):
    # MLflow 3.x puts the local file store in maintenance mode; opt in so the
    # `./mlruns` workflow keeps working without standing up a database backend.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = config.get("mlflow_tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri("file:///" + os.path.abspath(os.path.expanduser(tracking_uri)).replace("\\", "/"))
    mlflow.set_experiment(_cfg(config, "mlflow_experiment"))

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    with mlflow.start_run(run_name=datetime.now().strftime("run_%Y%m%d_%H%M%S")) as run:
        mlflow.log_params({
            "test_size": _cfg(config, "test_size"),
            "val_size": _cfg(config, "val_size"),
            "random_state": _cfg(config, "random_state"),
            "cv_n_splits": _cfg(config, "n_splits"),
            "n_train": len(split["X_train"]),
            "n_val": len(split["X_val"]),
            "n_test": len(split["X_test"]),
            "tfidf_ngram_range": str(tfidf.ngram_range),
            "tfidf_min_df": tfidf.min_df,
            "clf": clf.__class__.__name__,
            "clf_C": getattr(clf, "C", None),
        })
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        for i, score in enumerate(cv_scores):
            mlflow.log_metric("cv_fold_accuracy", float(score), step=i)

        mlflow.log_text(classification_report(split["y_test"], y_pred),
                        "classification_report.txt")
        cm = confusion_matrix(split["y_test"], y_pred)
        mlflow.log_text(json.dumps({"labels": [0, 1], "matrix": cm.tolist()}, indent=2),
                        "confusion_matrix.json")
        mlflow.log_text(
            pd.DataFrame({"y_test": split["y_test"], "y_pred": y_pred, "y_proba": y_proba}).to_csv(index=False),
            "roc_data.csv",
        )
        mlflow.sklearn.log_model(pipeline, name="model")
        logger.info("MLflow run %s logged", run.info.run_id)


def _save_artifacts(config, pipeline, split, pmids, labels, cv_metrics,
                    val_metrics, test_metrics, y_pred, y_proba):
    output_model = os.path.abspath(os.path.expanduser(
        config.get("output_model", "./output/model.pkl")))
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(pipeline, output_model)
    logger.info("saved model -> %s", output_model)

    roc_path = config.get("output_roc_data")
    if roc_path:
        roc_path = os.path.abspath(os.path.expanduser(roc_path))
        pd.DataFrame({"y_test": split["y_test"], "y_pred": y_pred,
                      "y_proba": y_proba}).to_csv(roc_path, index=False)

    pmids = np.asarray(pmids)
    labels = np.asarray(labels)
    trace_path = output_model.replace(".pkl", "_trace.csv")
    pd.DataFrame({
        "pmid": np.concatenate([pmids[split["idx_train"]], pmids[split["idx_val"]], pmids[split["idx_test"]]]),
        "class_label": np.concatenate([labels[split["idx_train"]], labels[split["idx_val"]], labels[split["idx_test"]]]),
        "split": (["train"] * len(split["idx_train"])
                  + ["validation"] * len(split["idx_val"])
                  + ["test"] * len(split["idx_test"])),
    }).to_csv(trace_path, index=False)

    report_path = output_model.replace(".pkl", "_training.log")
    with open(report_path, "w", encoding="utf-8") as log:
        log.write("REPORT FOR TRAINING PUBMED CLASSIFIER\n")
        log.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        log.write(f"split -> train {len(split['X_train'])} | "
                  f"val {len(split['X_val'])} | test {len(split['X_test'])}\n")
        log.write(f"cross-validation accuracy: {cv_metrics['cv_accuracy_mean']:.4f} "
                  f"± {cv_metrics['cv_accuracy_std']:.4f}\n")
        log.write(f"validation: {val_metrics}\n\n")
        log.write("test classification report:\n")
        log.write(classification_report(split["y_test"], y_pred))
        log.write(f"\ntest ROC AUC: {test_metrics['roc_auc']:.4f}\n")

    return {"model_path": output_model, "trace_path": trace_path, "report_path": report_path}
