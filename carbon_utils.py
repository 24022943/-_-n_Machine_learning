"""
carbon_utils.py
Core utilities for EcoPredict Carbon.

Đề tài: Ứng dụng học máy trong xây dựng hệ thống dự báo phát thải carbon của sản phẩm.
Dataset: The Carbon Catalogue - Product Level Data.

Điểm thiết kế chính:
- Đọc CSV an toàn với nhiều encoding.
- Chuẩn hóa tên cột Carbon Catalogue.
- Feature engineering có kiểm soát leakage.
- Tạo nhãn Low/Medium/High bằng ngưỡng train-only.
- Tạo sklearn Pipeline cho classification và regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import re
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

RANDOM_STATE = 42
TARGET_COL = "pcf_kg_co2e"
LABEL_COL = "carbon_label"
LABEL_NUM_COL = "carbon_label_num"
LABEL_ORDER = ["Low", "Medium", "High"]
LABEL_TO_NUM = {"Low": 0, "Medium": 1, "High": 2}
NUM_TO_LABEL = {v: k for k, v in LABEL_TO_NUM.items()}
LABEL_VI = {"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"}

COLUMN_MAP = {
    "*PCF-ID": "pcf_id",
    "PCF-ID": "pcf_id",
    "Year of reporting": "year",
    "*Stage-level CO2e available": "stage_level_available",
    "Stage-level CO2e available": "stage_level_available",
    "Product name (and functional unit)": "product_name",
    "Product detail": "product_detail",
    "Company": "company",
    "Country (where company is incorporated)": "country",
    "Company's GICS Industry Group": "industry_group",
    "Company's GICS Industry": "industry",
    "*Company's sector": "company_sector",
    "Company's sector": "company_sector",
    "Product weight (kg)": "product_weight_kg",
    "*Source for product weight": "weight_source",
    "Source for product weight": "weight_source",
    "Product's carbon footprint (PCF, kg CO2e)": "pcf_kg_co2e",
    "*Carbon intensity": "carbon_intensity",
    "Carbon intensity": "carbon_intensity",
    "Protocol used for PCF": "protocol",
    "Relative change in PCF vs previous": "relative_change_pcf",
    "Company-reported reason for change": "reason_for_change",
    "*Change reason category": "change_reason_category",
    "Change reason category": "change_reason_category",
    "*%Upstream estimated from %Operations": "upstream_estimated_from_operations",
    "%Upstream estimated from %Operations": "upstream_estimated_from_operations",
    "*Upstream CO2e (fraction of total PCF)": "upstream_frac",
    "Upstream CO2e (fraction of total PCF)": "upstream_frac",
    "*Operations CO2e (fraction of total PCF)": "operations_frac",
    "Operations CO2e (fraction of total PCF)": "operations_frac",
    "*Downstream CO2e (fraction of total PCF)": "downstream_frac",
    "Downstream CO2e (fraction of total PCF)": "downstream_frac",
    "*Transport CO2e (fraction of total PCF)": "transport_frac",
    "Transport CO2e (fraction of total PCF)": "transport_frac",
    "*EndOfLife CO2e (fraction of total PCF)": "end_of_life_frac",
    "EndOfLife CO2e (fraction of total PCF)": "end_of_life_frac",
    "*Adjustments to raw data (if any)": "raw_data_adjustments",
    "Adjustments to raw data (if any)": "raw_data_adjustments",
}

CORE_COLUMNS = [
    "year",
    "stage_level_available",
    "product_name",
    "product_detail",
    "company",
    "country",
    "industry_group",
    "industry",
    "company_sector",
    "product_weight_kg",
    "weight_source",
    TARGET_COL,
    "protocol",
    "upstream_estimated_from_operations",
    "upstream_frac",
    "operations_frac",
    "downstream_frac",
    "transport_frac",
    "end_of_life_frac",
]

LEAKAGE_COLUMNS = {
    TARGET_COL,
    "carbon_intensity",  # PCF / weight, should not be a predictor.
    LABEL_COL,
    LABEL_NUM_COL,
}

VI_FEATURE_NAMES = {
    "year": "Năm báo cáo",
    "product_weight_kg": "Khối lượng sản phẩm",
    "product_weight_log": "Khối lượng sản phẩm (log)",
    "country": "Quốc gia",
    "industry_group": "Nhóm ngành GICS",
    "industry": "Ngành sản phẩm",
    "company_sector": "Sector công ty",
    "protocol_simple": "Chuẩn PCF",
    "stage_level_available": "Có dữ liệu theo giai đoạn",
    "has_stage_data": "Có dữ liệu vòng đời",
    "weight_source": "Nguồn khối lượng",
    "is_weight_estimated": "Khối lượng ước tính",
    "upstream_estimated_flag": "Upstream được ước tính",
    "upstream_frac": "Tỷ trọng phát thải đầu vào",
    "operations_frac": "Tỷ trọng phát thải vận hành",
    "downstream_frac": "Tỷ trọng phát thải đầu ra",
    "transport_frac": "Tỷ trọng vận chuyển",
    "end_of_life_frac": "Tỷ trọng thải bỏ",
    "lifecycle_fraction_sum": "Tổng tỷ trọng vòng đời",
    "lifecycle_balance_std": "Độ lệch giữa các giai đoạn vòng đời",
    "dominant_stage": "Giai đoạn vòng đời chiếm ưu thế",
    "weight_category": "Nhóm khối lượng sản phẩm",
    "product_name_length": "Độ dài tên sản phẩm",
    "product_detail_length": "Độ dài mô tả sản phẩm",
    "upstream_x_weight_log": "Tương tác upstream và khối lượng",
    "operations_x_weight_log": "Tương tác vận hành và khối lượng",
    "downstream_x_weight_log": "Tương tác downstream và khối lượng",
}


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    """Read CSV by trying common encodings used in Carbon Catalogue exports."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    errors: list[str] = []
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{enc}: {exc}")
    raise ValueError("Không đọc được CSV. Thử lại encoding khác. Chi tiết: " + " | ".join(errors[:3]))


def percent_to_float(value: Any) -> float:
    """Convert messy percent/fraction strings to fraction in [0,1] where possible."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        x = float(value)
        if abs(x) > 1.5 and abs(x) <= 100:
            return x / 100.0
        return x
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "n/a", "na", "not reported", "blank"}:
        return np.nan
    if "included" in s.lower() or "insufficient" in s.lower() or "not reported" in s.lower():
        return np.nan
    nums = re.findall(r"[-+]?\d*\.?\d+", s.replace(",", ""))
    if not nums:
        return np.nan
    x = float(nums[0])
    if "%" in s or abs(x) > 1.5:
        return x / 100.0
    return x


def parse_number(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "n/a", "na", "not reported", "blank"}:
        return np.nan
    nums = re.findall(r"[-+]?\d*\.?\d+", s.replace(",", ""))
    if not nums:
        return np.nan
    return float(nums[0])


def normalize_yes_no(value: Any) -> str:
    s = str(value).strip().lower()
    if s.startswith("y"):
        return "Yes"
    if s.startswith("n"):
        return "No"
    return "Unknown"


def simplify_protocol(value: Any) -> str:
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "n/a", "na", "not reported", "(not reported by company)"}:
        return "Unknown"
    if "ghg" in s or "greenhouse" in s:
        return "GHG Protocol"
    if "iso" in s:
        return "ISO"
    if "pas" in s or "2050" in s:
        return "PAS 2050"
    return "Other"


def clean_category(value: Any, default: str = "Unknown") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "n/a", "na", "not used for 2015 reporting", "field not included in 2013 data"}:
        return default
    return s


def load_carbon_catalogue(path: str | Path) -> pd.DataFrame:
    """Load and standardize Carbon Catalogue product-level data."""
    raw = read_csv_safely(path)
    df = raw.rename(columns={c: COLUMN_MAP.get(c, c) for c in raw.columns}).copy()

    # Ensure expected columns exist.
    for col in CORE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = ["year", "product_weight_kg", TARGET_COL, "carbon_intensity"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].map(parse_number)

    frac_cols = ["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]
    for col in frac_cols:
        df[col] = df[col].map(percent_to_float)

    categorical_cols = [
        "stage_level_available",
        "product_name",
        "product_detail",
        "company",
        "country",
        "industry_group",
        "industry",
        "company_sector",
        "weight_source",
        "protocol",
        "upstream_estimated_from_operations",
    ]
    for col in categorical_cols:
        df[col] = df[col].map(clean_category)

    df["stage_level_available"] = df["stage_level_available"].map(normalize_yes_no)
    df["upstream_estimated_from_operations"] = df["upstream_estimated_from_operations"].map(normalize_yes_no)
    df["protocol_simple"] = df["protocol"].map(simplify_protocol)

    # Domain-valid filtering: PCF and product weight must be positive for supervised learning.
    df = df[(df[TARGET_COL] > 0) & (df["product_weight_kg"] > 0)].copy()
    df = df.drop_duplicates(subset=["pcf_id"], keep="first") if "pcf_id" in df.columns else df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return add_model_features(df)


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features without using target leakage variables."""
    out = df.copy()
    for col in ["product_weight_kg", "year", "upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]:
        if col not in out.columns:
            out[col] = np.nan

    out["product_weight_log"] = np.log1p(pd.to_numeric(out["product_weight_kg"], errors="coerce").clip(lower=0))
    out["has_stage_data"] = (out.get("stage_level_available", "Unknown").astype(str).str.lower() == "yes").astype(int)
    out["is_weight_estimated"] = out.get("weight_source", "").astype(str).str.lower().str.contains("estimated", na=False).astype(int)
    out["upstream_estimated_flag"] = (out.get("upstream_estimated_from_operations", "Unknown").astype(str).str.lower() == "yes").astype(int)
    out["product_name_length"] = out.get("product_name", "").astype(str).str.len().clip(0, 200)
    out["product_detail_length"] = out.get("product_detail", "").astype(str).str.len().clip(0, 500)

    lifecycle_cols = ["upstream_frac", "operations_frac", "downstream_frac"]
    for col in lifecycle_cols + ["transport_frac", "end_of_life_frac"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Fill missing lifecycle fractions softly for feature construction only.
    tmp = out[lifecycle_cols].copy()
    row_sum = tmp.sum(axis=1, skipna=True)
    missing_all = tmp.isna().all(axis=1)
    tmp = tmp.div(row_sum.replace(0, np.nan), axis=0)
    tmp.loc[missing_all, :] = np.nan
    tmp = tmp.fillna(1 / 3)
    out["lifecycle_fraction_sum"] = out[lifecycle_cols].sum(axis=1, skipna=True)
    out["lifecycle_balance_std"] = tmp.std(axis=1)

    dom_idx = tmp.idxmax(axis=1)
    out["dominant_stage"] = dom_idx.map({
        "upstream_frac": "Upstream",
        "operations_frac": "Operations",
        "downstream_frac": "Downstream",
    }).fillna("Unknown")

    # Weight category without target information.
    q1, q2 = out["product_weight_kg"].quantile([0.33, 0.66]).values if out["product_weight_kg"].notna().sum() >= 3 else (1, 10)
    def weight_bin(x: float) -> str:
        if pd.isna(x):
            return "Unknown"
        if x <= q1:
            return "Light"
        if x <= q2:
            return "Medium"
        return "Heavy"
    out["weight_category"] = out["product_weight_kg"].map(weight_bin)

    out["upstream_x_weight_log"] = tmp["upstream_frac"] * out["product_weight_log"]
    out["operations_x_weight_log"] = tmp["operations_frac"] * out["product_weight_log"]
    out["downstream_x_weight_log"] = tmp["downstream_frac"] * out["product_weight_log"]

    return out


def fit_label_thresholds(y_train_pcf: pd.Series | np.ndarray) -> dict[str, float]:
    """Fit Low/Medium/High thresholds on TRAIN target only to avoid leakage."""
    y = pd.Series(y_train_pcf).astype(float)
    q25 = float(y.quantile(0.25))
    q75 = float(y.quantile(0.75))
    return {"q25": q25, "q75": q75}


def apply_carbon_labels(y_pcf: pd.Series | np.ndarray, thresholds: dict[str, float]) -> pd.Series:
    y = pd.Series(y_pcf).astype(float)
    labels = np.where(y <= thresholds["q25"], "Low", np.where(y >= thresholds["q75"], "High", "Medium"))
    return pd.Series(labels, index=y.index)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return all feature columns, numeric features and categorical features."""
    numeric_features = [
        "year",
        "product_weight_kg",
        "product_weight_log",
        "upstream_frac",
        "operations_frac",
        "downstream_frac",
        "transport_frac",
        "end_of_life_frac",
        "has_stage_data",
        "is_weight_estimated",
        "upstream_estimated_flag",
        "product_name_length",
        "product_detail_length",
        "lifecycle_fraction_sum",
        "lifecycle_balance_std",
        "upstream_x_weight_log",
        "operations_x_weight_log",
        "downstream_x_weight_log",
    ]
    categorical_features = [
        "country",
        "industry_group",
        "industry",
        "company_sector",
        "protocol_simple",
        "stage_level_available",
        "weight_source",
        "dominant_stage",
        "weight_category",
    ]
    numeric_features = [c for c in numeric_features if c in df.columns and c not in LEAKAGE_COLUMNS]
    categorical_features = [c for c in categorical_features if c in df.columns and c not in LEAKAGE_COLUMNS]
    feature_cols = numeric_features + categorical_features
    return feature_cols, numeric_features, categorical_features


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str], robust: bool = True) -> ColumnTransformer:
    scaler = RobustScaler() if robust else StandardScaler()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_ohe()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ], remainder="drop", verbose_feature_names_out=False)


def make_classifier_candidates(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Dummy Baseline": Pipeline([("prep", preprocessor), ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))]),
        "Logistic Regression": Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(n_estimators=350, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE))]),
        "Extra Trees": Pipeline([("prep", preprocessor), ("model", ExtraTreesClassifier(n_estimators=450, max_depth=None, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE))]),
        "Hist Gradient Boosting": Pipeline([("prep", preprocessor), ("model", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, l2_regularization=0.05, random_state=RANDOM_STATE))]),
    }


def make_regressor_candidates(preprocessor: ColumnTransformer) -> dict[str, Any]:
    base = {
        "Dummy Mean": Pipeline([("prep", preprocessor), ("model", DummyRegressor(strategy="mean"))]),
        "Ridge": Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))]),
        "Random Forest Regressor": Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=350, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))]),
        "Extra Trees Regressor": Pipeline([("prep", preprocessor), ("model", ExtraTreesRegressor(n_estimators=450, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))]),
        "Hist Gradient Boosting Regressor": Pipeline([("prep", preprocessor), ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, l2_regularization=0.05, random_state=RANDOM_STATE))]),
    }
    out: dict[str, Any] = {}
    for name, pipe in base.items():
        if name == "Dummy Mean":
            out[name] = pipe
        else:
            out[name] = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    return out


def param_distributions_for_classifier(name: str) -> Optional[dict[str, list[Any]]]:
    if name == "Logistic Regression":
        return {"model__C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]}
    if name == "Random Forest":
        return {
            "model__n_estimators": [250, 350, 500],
            "model__max_depth": [None, 8, 14, 22],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        }
    if name == "Extra Trees":
        return {
            "model__n_estimators": [300, 450, 650],
            "model__max_depth": [None, 8, 14, 22],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        }
    if name == "Hist Gradient Boosting":
        return {
            "model__max_iter": [160, 250, 350],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 0.05, 0.2],
        }
    return None


def tune_classifier_if_needed(name: str, estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 12) -> Pipeline:
    params = param_distributions_for_classifier(name)
    if not params:
        estimator.fit(X_train, y_train)
        return estimator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=params,
        n_iter=min(n_iter, max(1, int(np.prod([len(v) for v in params.values()])))),
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    best.best_params_ = search.best_params_  # type: ignore[attr-defined]
    best.best_cv_score_ = float(search.best_score_)  # type: ignore[attr-defined]
    return best


def evaluate_classifier(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    row: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABEL_ORDER),
        "classification_report": classification_report(y_test, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0),
    }
    try:
        proba = model.predict_proba(X_test)
        row["roc_auc_ovr"] = roc_auc_score(y_test, proba, labels=LABEL_ORDER, multi_class="ovr", average="macro")
        row["proba"] = proba
    except Exception:  # noqa: BLE001
        row["roc_auc_ovr"] = np.nan
        row["proba"] = None
    return row


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_pred_arr = np.maximum(y_pred_arr, 0)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = math.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)
    ape = np.abs(y_pred_arr - y_true_arr) / np.maximum(np.abs(y_true_arr), 1e-9) * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        rmsle = math.sqrt(mean_squared_error(np.log1p(np.maximum(y_true_arr, 0)), np.log1p(np.maximum(y_pred_arr, 0))))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "rmsle": float(rmsle),
    }


def leakage_screening(df: pd.DataFrame, feature_cols: list[str], target_col: str = TARGET_COL) -> pd.DataFrame:
    """Rank numeric features by absolute Spearman correlation with target."""
    rows = []
    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            try:
                corr = df[[col, target_col]].corr(method="spearman").iloc[0, 1]
            except Exception:  # noqa: BLE001
                corr = np.nan
            rows.append({"feature": col, "spearman_abs_corr_with_target": abs(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows).sort_values("spearman_abs_corr_with_target", ascending=False)


def save_package(package: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)


def load_package(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def get_available_csv() -> Path:
    for name in ["carbon_catalogue.csv", "PublicTablesForCarbonCatalogueDataDescriptor_v30Oct2021(Product Level Data).csv"]:
        p = Path(name)
        if p.exists():
            return p
    raise FileNotFoundError("Không tìm thấy carbon_catalogue.csv trong thư mục hiện tại.")


def translate_feature_name(name: str) -> str:
    if name in VI_FEATURE_NAMES:
        return VI_FEATURE_NAMES[name]
    # One-hot encoded names may look like: industry_group_Automobiles & Components
    for prefix, vi in VI_FEATURE_NAMES.items():
        key = prefix + "_"
        if name.startswith(key):
            return f"{vi}: {name[len(key):]}"
    clean = name.replace("__", "_").replace("_", " ")
    return clean[:1].upper() + clean[1:]


def shorten_label(text: str, max_len: int = 34) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def build_input_row(reference_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Build a single-row dataframe compatible with the trained feature pipeline."""
    row: dict[str, Any] = {}
    for col in reference_df.columns:
        if col in kwargs:
            row[col] = kwargs[col]
        elif col in ["product_name", "product_detail"]:
            row[col] = "User input product"
        elif col in ["company"]:
            row[col] = "Unknown"
        elif col == TARGET_COL:
            row[col] = np.nan
        elif col in ["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]:
            row[col] = np.nan
        elif pd.api.types.is_numeric_dtype(reference_df[col]):
            row[col] = float(reference_df[col].median()) if reference_df[col].notna().any() else 0.0
        else:
            mode = reference_df[col].mode(dropna=True)
            row[col] = str(mode.iloc[0]) if len(mode) else "Unknown"
    row.update(kwargs)
    return add_model_features(pd.DataFrame([row]))
