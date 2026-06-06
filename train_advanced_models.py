"""
train_advanced_models.py
Train improved EcoPredict Carbon ML models.

Chạy:
    python train_advanced_models.py

Đầu ra:
    outputs/models/ecopredict_model_package.joblib
    outputs/tables/*.csv
    outputs/figures/*.png
    outputs/carbon_processed_advanced.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import math
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import label_binarize

from carbon_utils import (
    LABEL_ORDER,
    LABEL_TO_NUM,
    NUM_TO_LABEL,
    RANDOM_STATE,
    TARGET_COL,
    apply_carbon_labels,
    build_preprocessor,
    evaluate_classifier,
    fit_label_thresholds,
    get_feature_columns,
    leakage_screening,
    load_carbon_catalogue,
    make_classifier_candidates,
    make_regressor_candidates,
    regression_metrics,
    save_package,
    translate_feature_name,
    tune_classifier_if_needed,
)

warnings.filterwarnings("ignore")

DATA_PATH = Path("carbon_catalogue.csv")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"
MODEL_PATH = MODEL_DIR / "ecopredict_model_package.joblib"


def ensure_dirs() -> None:
    for p in [OUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def save_classification_report(report: dict[str, Any], model_name: str) -> None:
    rows = []
    for label in LABEL_ORDER + ["macro avg", "weighted avg"]:
        if label in report:
            row = {"label": label, **report[label]}
            rows.append(row)
    pd.DataFrame(rows).to_csv(TABLE_DIR / f"classification_report_{model_name.replace(' ', '_').lower()}.csv", index=False, encoding="utf-8-sig")


def plot_confusion_matrix(best_model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, labels=LABEL_ORDER, cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix_best_classifier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(classifier_results: dict[str, dict[str, Any]], y_test: pd.Series) -> None:
    y_bin = label_binarize(y_test, classes=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for name, res in classifier_results.items():
        proba = res.get("proba")
        if proba is None:
            continue
        # Micro-average multiclass ROC.
        fpr, tpr, _ = roc_curve(y_bin.ravel(), np.asarray(proba).ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison - Test Set")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(best_model: Any, X_test: pd.DataFrame, y_test: pd.Series, positive_label: str = "High") -> None:
    """Calibration for one-vs-rest probability of High carbon class."""
    try:
        proba = best_model.predict_proba(X_test)
        cls = list(best_model.classes_)
        idx = cls.index(positive_label)
        y_bin = (y_test == positive_label).astype(int).to_numpy()
        frac_pos, mean_pred = calibration_curve(y_bin, proba[:, idx], n_bins=8, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{positive_label} one-vs-rest")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve - High carbon probability")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "calibration_curve_high_class.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"Không vẽ được calibration curve: {exc}")


def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(y_test, y_pred, alpha=0.6, s=28)
    max_val = max(float(np.nanmax(y_test)), float(np.nanmax(y_pred)))
    ax.plot([0, max_val], [0, max_val], "--", color="gray", lw=1.5)
    ax.set_xlabel("Actual PCF (kg CO₂e)")
    ax.set_ylabel("Predicted PCF (kg CO₂e)")
    ax.set_title(f"Predicted vs Actual - {model_name}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regression_predicted_vs_actual.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residual_distribution(y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    residual = np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(residual, bins=28, alpha=0.78, edgecolor="white")
    ax.axvline(0, linestyle="--", color="gray", lw=1.5)
    ax.set_xlabel("Residual = Actual - Predicted")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution - {model_name}")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regression_residual_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve_placeholder(summary: pd.DataFrame) -> None:
    # Lightweight figure: CV F1 by model, with std error bars.
    df = summary.sort_values("cv_f1_macro_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(df["model"], df["cv_f1_macro_mean"], xerr=df["cv_f1_macro_std"], alpha=0.85)
    ax.set_xlabel("5-Fold CV F1-macro")
    ax.set_title("Model Comparison by Cross-Validation")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv_f1_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_importance(best_model: Any, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    print("Đang tính Permutation Importance...")
    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": feature_cols,
        "feature_vi": [translate_feature_name(c) for c in feature_cols],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(TABLE_DIR / "permutation_importance_best_classifier.csv", index=False, encoding="utf-8-sig")

    top = imp.head(14).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.barh(top["feature_vi"], top["importance_mean"], xerr=top["importance_std"], alpha=0.85)
    ax.set_xlabel("Decrease in F1-macro after permutation")
    ax.set_title(f"Permutation Importance - {model_name}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_importance_best_classifier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return imp


def train_classifiers(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, preprocessor: Any, feature_cols: list[str]) -> tuple[str, Any, pd.DataFrame, dict[str, dict[str, Any]]]:
    candidates = make_classifier_candidates(preprocessor)
    rows = []
    results: dict[str, dict[str, Any]] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in candidates.items():
        print(f"\n[Classification] Training {name}...")
        if name in ["Dummy Baseline"]:
            fitted = model.fit(X_train, y_train)
        else:
            fitted = tune_classifier_if_needed(name, model, X_train, y_train, n_iter=12)
        res = evaluate_classifier(fitted, X_test, y_test)
        try:
            cv_scores = cross_val_score(fitted, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1)
        except Exception:
            cv_scores = np.array([np.nan])
        best_params = getattr(fitted, "best_params_", {})
        row = {
            "model": name,
            "accuracy": res["accuracy"],
            "balanced_accuracy": res["balanced_accuracy"],
            "f1_macro": res["f1_macro"],
            "f1_weighted": res["f1_weighted"],
            "roc_auc_ovr": res["roc_auc_ovr"],
            "cv_f1_macro_mean": float(np.nanmean(cv_scores)),
            "cv_f1_macro_std": float(np.nanstd(cv_scores)),
            "best_params": json.dumps(best_params, ensure_ascii=False),
        }
        rows.append(row)
        res["estimator"] = fitted
        results[name] = res
        save_classification_report(res["classification_report"], name)
        print(pd.Series(row).drop(labels=["best_params"]).to_string())

    summary = pd.DataFrame(rows).sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
    summary.to_csv(TABLE_DIR / "classification_model_comparison.csv", index=False, encoding="utf-8-sig")
    plot_learning_curve_placeholder(summary)
    best_name = str(summary.iloc[0]["model"])
    best_model = results[best_name]["estimator"]
    print(f"\nBest classifier: {best_name}")
    plot_confusion_matrix(best_model, X_test, y_test, best_name)
    plot_roc_curves(results, y_test)
    plot_calibration(best_model, X_test, y_test, positive_label="High")
    plot_permutation_importance(best_model, X_test, y_test, feature_cols, best_name)
    return best_name, best_model, summary, results


def train_regressors(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, preprocessor: Any) -> tuple[str, Any, pd.DataFrame, dict[str, Any]]:
    candidates = make_regressor_candidates(preprocessor)
    rows = []
    fitted_models: dict[str, Any] = {}

    for name, model in candidates.items():
        print(f"\n[Regression] Training {name}...")
        fitted = model.fit(X_train, y_train)
        y_pred = np.maximum(fitted.predict(X_test), 0)
        met = regression_metrics(y_test, y_pred)
        row = {"model": name, **met}
        rows.append(row)
        fitted_models[name] = fitted
        print(pd.Series(row).to_string())

    summary = pd.DataFrame(rows)
    # Composite selection: prioritize lower Median APE, then lower MAE, then higher R2.
    summary["rank_median_ape"] = summary["median_ape"].rank(method="min", ascending=True)
    summary["rank_mae"] = summary["mae"].rank(method="min", ascending=True)
    summary["rank_r2"] = summary["r2"].rank(method="min", ascending=False)
    summary["selection_score"] = summary["rank_median_ape"] + summary["rank_mae"] + 0.5 * summary["rank_r2"]
    summary = summary.sort_values("selection_score", ascending=True)
    summary.to_csv(TABLE_DIR / "regression_model_comparison.csv", index=False, encoding="utf-8-sig")
    best_name = str(summary.iloc[0]["model"])
    best_model = fitted_models[best_name]
    y_pred = np.maximum(best_model.predict(X_test), 0)
    plot_predicted_vs_actual(y_test, y_pred, best_name)
    plot_residual_distribution(y_test, y_pred, best_name)
    print(f"\nBest regressor: {best_name}")
    return best_name, best_model, summary, fitted_models


def main(data_path: str | Path = DATA_PATH) -> dict[str, Any]:
    ensure_dirs()
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {data_path}. Hãy đặt file dữ liệu tên carbon_catalogue.csv cùng thư mục.")

    print("=" * 80)
    print("ECOPREDICT CARBON - ADVANCED TRAINING")
    print("=" * 80)

    df = load_carbon_catalogue(data_path)
    feature_cols, numeric_features, categorical_features = get_feature_columns(df)

    if TARGET_COL in feature_cols or "carbon_intensity" in feature_cols:
        raise ValueError("Feature set còn chứa target/leakage columns. Kiểm tra get_feature_columns().")

    df.to_csv(OUT_DIR / "carbon_processed_advanced.csv", index=False, encoding="utf-8-sig")

    X = df[feature_cols].copy()
    y_pcf = df[TARGET_COL].astype(float)

    X_train, X_test, y_train_pcf, y_test_pcf = train_test_split(
        X,
        y_pcf,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    thresholds = fit_label_thresholds(y_train_pcf)
    y_train_label = apply_carbon_labels(y_train_pcf, thresholds)
    y_test_label = apply_carbon_labels(y_test_pcf, thresholds)

    # Keep label distribution tables for report.
    pd.DataFrame({
        "split": ["train"] * len(y_train_label) + ["test"] * len(y_test_label),
        "label": list(y_train_label) + list(y_test_label),
    }).groupby(["split", "label"]).size().reset_index(name="count").to_csv(TABLE_DIR / "label_distribution_train_test.csv", index=False, encoding="utf-8-sig")

    leakage = leakage_screening(pd.concat([X, y_pcf.rename(TARGET_COL)], axis=1), feature_cols, TARGET_COL)
    leakage.to_csv(TABLE_DIR / "leakage_screening_numeric_features.csv", index=False, encoding="utf-8-sig")

    preprocessor = build_preprocessor(numeric_features, categorical_features, robust=True)

    best_clf_name, best_clf, clf_summary, clf_results = train_classifiers(
        X_train, X_test, y_train_label, y_test_label, preprocessor, feature_cols
    )

    best_reg_name, best_reg, reg_summary, reg_models = train_regressors(
        X_train, X_test, y_train_pcf, y_test_pcf, preprocessor
    )

    package = {
        "classifier": best_clf,
        "classifier_name": best_clf_name,
        "regressor": best_reg,
        "regressor_name": best_reg_name,
        "thresholds": thresholds,
        "metadata": {
            "feature_cols": feature_cols,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "label_order": LABEL_ORDER,
            "target_col": TARGET_COL,
            "random_state": RANDOM_STATE,
            "split": "80/20 train-test",
            "label_method": "Q25/Q75 fitted on y_train only to avoid leakage",
        },
        "reference_data": df,
        "classification_summary": clf_summary,
        "regression_summary": reg_summary,
        "y_test_pcf": y_test_pcf,
        "y_test_label": y_test_label,
        "X_test": X_test,
    }
    save_package(package, MODEL_PATH)
    print("\n" + "=" * 80)
    print(f"Đã lưu model package: {MODEL_PATH}")
    print(f"Đã lưu dữ liệu xử lý: {OUT_DIR / 'carbon_processed_advanced.csv'}")
    print(f"Đã lưu bảng kết quả: {TABLE_DIR}")
    print(f"Đã lưu biểu đồ: {FIG_DIR}")
    print("=" * 80)
    return package


if __name__ == "__main__":
    main()
