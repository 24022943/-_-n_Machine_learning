"""
app.py
Streamlit web app for EcoPredict Carbon.

Chạy:
    python -m streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from carbon_utils import (
    LABEL_ORDER,
    LABEL_TO_NUM,
    LABEL_VI,
    TARGET_COL,
    build_input_row,
    get_feature_columns,
    load_carbon_catalogue,
    load_package,
    shorten_label,
    translate_feature_name,
)

DATA_PATH = Path("carbon_catalogue.csv")
from pathlib import Path
import joblib
import streamlit as st

MODEL_CANDIDATES = [
    Path("ecopredict_model_package.joblib"),
    Path("outputs/models/ecopredict_model_package.joblib"),
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

if MODEL_PATH is None:
    st.error(
        "Không tìm thấy file ecopredict_model_package.joblib. "
        "Hãy upload file model này lên GitHub cùng cấp với app.py."
    )
    st.stop()

package = joblib.load(MODEL_PATH)
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(
    page_title="EcoPredict Carbon",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --primary:#15803d; --primary-dark:#064e3b; --accent:#10b981;
  --bg:#f5fbf7; --card:#ffffff; --muted:#667085; --line:#dbe7e0;
}
[data-testid="stAppViewContainer"] {background: linear-gradient(180deg,#f8fcfa 0%,#eef7f2 100%);}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#063b2d 0%,#0f5132 100%);}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#f7fff9 !important;}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] div[data-baseweb="input"] input,
[data-testid="stSidebar"] div[data-baseweb="base-input"] input,
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="select"] span {
  color:#111827 !important; -webkit-text-fill-color:#111827 !important;
}
[data-testid="stSidebar"] input::placeholder {color:#6b7280 !important; -webkit-text-fill-color:#6b7280 !important; opacity:1 !important;}
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] > div,
[data-testid="stSidebar"] div[data-baseweb="base-input"] {
  background:#fff !important; border-radius:14px !important;
}
div[data-baseweb="popover"], div[data-baseweb="popover"] *, ul[data-testid="stVirtualDropdown"], ul[data-testid="stVirtualDropdown"] *, div[role="listbox"], div[role="option"] {
  color:#111827 !important; -webkit-text-fill-color:#111827 !important;
}
.hero-shell {
  position:relative; overflow:hidden; margin-bottom:24px; padding:38px 42px 30px 42px;
  border-radius:36px; background:linear-gradient(180deg,#f4fbf7 0%,#edf8f2 100%);
  border:1px solid #bce5cf; box-shadow:0 18px 42px rgba(15,81,50,.08);
}
.hero-shell:after {content:""; position:absolute; right:-70px; bottom:-100px; width:390px; height:390px; border-radius:50%; background:radial-gradient(circle, rgba(16,185,129,.22), rgba(16,185,129,.08) 56%, transparent 57%);}
.hero-badge {display:inline-flex; gap:10px; align-items:center; padding:14px 24px; border-radius:999px; background:#d9ece4; color:#0f5132; font-size:18px; font-weight:850; margin-bottom:24px;}
.hero-title {position:relative; z-index:1; max-width:1180px; font-size:42px; line-height:1.13; font-weight:950; letter-spacing:-1.2px; color:#0f172a; margin:0 0 18px 0;}
.hero-subtitle {position:relative; z-index:1; max-width:1120px; font-size:18px; line-height:1.65; color:#667085; margin:0 0 24px 0;}
.hero-chip-row {display:flex; flex-wrap:wrap; gap:16px; position:relative; z-index:1;}
.hero-chip {display:inline-flex; align-items:center; padding:14px 24px; border-radius:999px; background:#d6efe3; color:#0f5132; font-size:16px; font-weight:850;}
.card {background:#fff; padding:24px; border-radius:28px; border:1px solid #e6efe9; box-shadow:0 12px 32px rgba(15,81,50,.07); margin-bottom:18px;}
.section-title {font-size:24px; font-weight:900; color:#0f172a; margin-bottom:6px;}
.card-subtitle {font-size:15px; color:#667085; margin-bottom:16px; line-height:1.6;}
.kpi-card {background:#fff; border-radius:26px; padding:24px; min-height:138px; border:1px solid #e8f2ed; box-shadow:0 12px 28px rgba(15,81,50,.07);}
.kpi-green {background:radial-gradient(circle at 90% 20%, rgba(16,185,129,.32), transparent 30%), linear-gradient(135deg,#067a3d,#0f5132); color:white; border:none;}
.kpi-orange {background:linear-gradient(135deg,#b45309,#f59e0b); color:white; border:none;}
.kpi-red {background:linear-gradient(135deg,#b91c1c,#f97316); color:white; border:none;}
.kpi-title {font-size:14px; color:#6b7280; font-weight:800; margin-bottom:10px;}
.kpi-value {font-size:40px; font-weight:950; letter-spacing:-1px; margin:0; color:#0f172a;}
.kpi-note {font-size:13px; color:#667085; line-height:1.5; margin-top:8px;}
.kpi-green .kpi-title,.kpi-green .kpi-note,.kpi-orange .kpi-title,.kpi-orange .kpi-note,.kpi-red .kpi-title,.kpi-red .kpi-note {color:rgba(255,255,255,.86);}
.kpi-green .kpi-value,.kpi-orange .kpi-value,.kpi-red .kpi-value {color:white;}
.success-box {padding:16px 18px; border-radius:18px; background:#ecfdf5; color:#064e3b; border:1px solid #a7f3d0; line-height:1.75; font-size:15.5px;}
.warning-box {padding:14px 16px; border-radius:16px; background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; line-height:1.6;}
.insight-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:4px 0 14px 0;}
.insight-card {background:#f8fbfa; border:1px solid #e5ece8; border-radius:18px; padding:16px 18px;}
.insight-label {font-size:13px; color:#64748b; font-weight:800; margin-bottom:6px;}
.insight-value {font-size:25px; color:#111827; font-weight:950; line-height:1.15;}
.insight-note {font-size:13px; color:#64748b; margin-top:4px; line-height:1.45;}
.explain-note {padding:16px 18px; border-radius:18px; background:#ecfdf5; color:#064e3b; border:1px solid #a7f3d0; font-size:15.5px; line-height:1.8; margin-bottom:12px;}
.small-muted {font-size:13px; color:#64748b; line-height:1.6;}
.badge {display:inline-block; padding:8px 12px; border-radius:999px; background:#dcfce7; color:#166534; font-size:13px; font-weight:850; margin:3px;}
@media (max-width:900px){.hero-title{font-size:31px}.insight-grid{grid-template-columns:1fr}.hero-shell{padding:28px}.kpi-value{font-size:34px}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def fmt_num(x: float, digits: int = 2) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


@st.cache_data(show_spinner=False)
def load_data_cached(path: str) -> pd.DataFrame:
    return load_carbon_catalogue(path)


@st.cache_resource(show_spinner=False)
def load_model_cached() -> dict[str, Any] | None:
    model_candidates = [
        Path("ecopredict_model_package.joblib"),
        Path("outputs/models/ecopredict_model_package.joblib"),
    ]

    for model_path in model_candidates:
        if model_path.exists():
            return load_package(str(model_path))

    st.error("Không tìm thấy model. Các file hiện có trong thư mục app:")
    st.write([p.name for p in Path(".").iterdir()])
    return None


def render_hero() -> None:
    st.markdown(
        """
        <div class='hero-shell'>
            <div class='hero-badge'>🌱 EcoPredict Carbon • VI / EN</div>
            <div class='hero-title'>EcoPredict Carbon – Hệ thống dự báo phát thải carbon của sản phẩm</div>
            <div class='hero-subtitle'>Product Carbon Footprint prediction and emission-level classification based on Carbon Catalogue data</div>
            <div class='hero-chip-row'>
                <div class='hero-chip'>Machine Learning</div>
                <div class='hero-chip'>Carbon Catalogue</div>
                <div class='hero-chip'>PCF Prediction</div>
                <div class='hero-chip'>Eco-Tech Dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_probability_bar(proba: np.ndarray) -> go.Figure:
    labels_vi = [LABEL_VI[x] for x in LABEL_ORDER]
    fig = go.Figure(go.Bar(x=labels_vi, y=proba * 100, marker_color=["#047857", "#f59e0b", "#ef4444"], text=[f"{p*100:.1f}%" for p in proba], textposition="outside", showlegend=False, name=""))
    fig.update_layout(height=310, margin=dict(l=20, r=20, t=10, b=40), yaxis_title="Xác suất (%)", xaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), yaxis=dict(range=[0, max(100, float(proba.max()*120))], gridcolor="#e5e7eb"), showlegend=False)
    return fig


def plot_lifecycle_donut(up: float, op: float, down: float) -> go.Figure:
    fig = go.Figure(go.Pie(labels=["Upstream", "Operations", "Downstream"], values=[up, op, down], hole=0.58, marker_colors=["#047857", "#10b981", "#86efac"], textinfo="label+percent", sort=False, showlegend=True))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), legend=dict(orientation="h", y=-0.08, x=0.1))
    return fig


def plot_benchmark(pred_pcf: float, industry_median: float, global_median: float) -> go.Figure:
    labels = ["Sản phẩm của bạn", "Trung vị ngành", "Trung vị toàn bộ"]
    values = [pred_pcf, industry_median, global_median]
    colors = ["#047857", "#64748b", "#cbd5e1"]
    fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", marker_color=colors, text=[fmt_num(v) for v in values], textposition="outside", hovertemplate="%{y}: %{x:,.2f} kg CO₂e<extra></extra>", showlegend=False, name=""))
    fig.update_layout(height=310, margin=dict(l=140, r=70, t=8, b=40), xaxis_title="kg CO₂e", yaxis_title="", showlegend=False, font=dict(color="#111827", size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(gridcolor="#e5e7eb", rangemode="tozero"), yaxis=dict(categoryorder="array", categoryarray=labels[::-1], automargin=True))
    return fig


def build_benchmark_analysis(df: pd.DataFrame, pred_pcf: float, industry_group: str, industry_median: float, global_median: float) -> tuple[str, float, float, int]:
    same = df[df["industry_group"].eq(industry_group)][TARGET_COL].dropna().astype(float)
    if len(same) < 3:
        same = df[TARGET_COL].dropna().astype(float)
    n_ref = int(len(same))
    industry_pct = float((same <= pred_pcf).mean() * 100) if n_ref else np.nan
    global_pct = float((df[TARGET_COL].dropna().astype(float) <= pred_pcf).mean() * 100)
    ratio = pred_pcf / max(industry_median, 1e-9)
    if ratio < 0.85:
        text = f"PCF dự báo thấp hơn đáng kể so với trung vị ngành ({fmt_num(industry_median)} kg CO₂e). Đây là tín hiệu tích cực, nhưng vẫn nên đối chiếu với dữ liệu LCA thực tế trước khi dùng như chứng nhận xanh."
    elif ratio <= 1.15:
        text = f"PCF dự báo nằm gần vùng trung vị ngành ({fmt_num(industry_median)} kg CO₂e). Sản phẩm có mức phát thải tương đối điển hình trong nhóm tham chiếu."
    else:
        text = f"PCF dự báo cao hơn trung vị ngành ({fmt_num(industry_median)} kg CO₂e). Nên rà soát khối lượng, vật liệu, nguồn năng lượng và phân bổ vòng đời để tìm cơ hội giảm phát thải."
    return text, industry_pct, global_pct, n_ref


def get_local_factor_effect(package: dict[str, Any], input_row: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    clf = package["classifier"]
    feature_cols = package["metadata"]["feature_cols"]
    base_proba = clf.predict_proba(input_row[feature_cols])[0]
    pred_idx = int(np.argmax(base_proba))
    pred_label = clf.classes_[pred_idx]
    base_score = float(base_proba[pred_idx])
    rows = []
    for col in feature_cols:
        perturbed = input_row.copy()
        if col not in perturbed.columns or col not in reference_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            replacement = float(reference_df[col].median()) if reference_df[col].notna().any() else 0.0
        else:
            mode = reference_df[col].mode(dropna=True)
            replacement = mode.iloc[0] if len(mode) else "Unknown"
        perturbed[col] = replacement
        try:
            new_score = float(clf.predict_proba(perturbed[feature_cols])[0][pred_idx])
            impact = base_score - new_score
            rows.append({"feature": col, "feature_vi": translate_feature_name(col), "impact": impact, "abs_impact": abs(impact)})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("abs_impact", ascending=False)


def plot_factor_effect(exp: pd.DataFrame) -> go.Figure:
    top = exp.head(6).copy()
    top["short"] = top["feature_vi"].map(lambda x: shorten_label(x, 32))
    top = top.sort_values("impact", ascending=True)
    max_abs = max(float(top["impact"].abs().max()), 1e-6)
    colors = np.where(top["impact"] >= 0, "#047857", "#f97316")
    fig = go.Figure(go.Bar(x=top["impact"], y=top["short"], orientation="h", marker_color=colors, text=[f"{v:+.3f}" for v in top["impact"]], textposition="outside", cliponaxis=False, customdata=np.stack([top["feature_vi"], top["impact"]], axis=1), hovertemplate="<b>%{customdata[0]}</b><br>Tác động: %{x:.4f}<extra></extra>", showlegend=False, name=""))
    fig.add_vline(x=0, line_width=1.2, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(height=420, margin=dict(l=175, r=90, t=10, b=44), xaxis_title="Mức tác động đến nhãn dự báo", yaxis_title="", showlegend=False, font=dict(color="#111827", size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[-max_abs*1.55, max_abs*1.55], zeroline=False, gridcolor="#e5e7eb"), yaxis=dict(automargin=True, tickfont=dict(size=12)))
    return fig


def sidebar_inputs(data: pd.DataFrame) -> dict[str, Any]:
    with st.sidebar:
        st.markdown("## 🌱 EcoPredict Carbon")
        st.caption("Nhập thông tin sản phẩm để dự báo PCF và phân loại mức phát thải.")
        st.markdown("---")
        year_min, year_max = int(data["year"].min()), int(data["year"].max())
        year = st.slider("Năm báo cáo", year_min, year_max, int(data["year"].median()))
        weight = st.number_input("Khối lượng sản phẩm (kg)", min_value=0.01, value=float(data["product_weight_kg"].median()), step=0.1)
        country = st.selectbox("Quốc gia", sorted(data["country"].dropna().unique().tolist()))
        industry_group = st.selectbox("Nhóm ngành GICS", sorted(data["industry_group"].dropna().unique().tolist()))
        industry_options = sorted(data.loc[data["industry_group"].eq(industry_group), "industry"].dropna().unique().tolist())
        if not industry_options:
            industry_options = sorted(data["industry"].dropna().unique().tolist())
        industry = st.selectbox("Ngành sản phẩm", industry_options)
        sector_options = sorted(data.loc[data["industry_group"].eq(industry_group), "company_sector"].dropna().unique().tolist())
        if not sector_options:
            sector_options = sorted(data["company_sector"].dropna().unique().tolist())
        sector = st.selectbox("Sector", sector_options)
        protocol = st.selectbox("Chuẩn PCF", sorted(data["protocol_simple"].dropna().unique().tolist()))
        weight_source = st.selectbox("Nguồn khối lượng", sorted(data["weight_source"].dropna().unique().tolist()))
        st.markdown("### Tỷ trọng vòng đời")
        upstream = st.slider("Upstream (%)", 0, 100, 45)
        operations = st.slider("Operations (%)", 0, 100, 35)
        downstream = st.slider("Downstream (%)", 0, 100, max(0, 100 - upstream - operations))
        total = upstream + operations + downstream
        if total != 100:
            st.warning(f"Tổng hiện tại = {total}%. Hệ thống sẽ tự chuẩn hóa khi dự báo.")
    total = max(total, 1)
    return {
        "year": year,
        "product_weight_kg": weight,
        "country": country,
        "industry_group": industry_group,
        "industry": industry,
        "company_sector": sector,
        "protocol_simple": protocol,
        "protocol": protocol,
        "weight_source": weight_source,
        "stage_level_available": "Yes",
        "upstream_estimated_from_operations": "No",
        "upstream_frac": upstream / total,
        "operations_frac": operations / total,
        "downstream_frac": downstream / total,
        "transport_frac": np.nan,
        "end_of_life_frac": np.nan,
    }


def prediction_page(data: pd.DataFrame, package: dict[str, Any]) -> None:
    render_hero()
    inputs = sidebar_inputs(data)
    clf = package["classifier"]
    reg = package["regressor"]
    feature_cols = package["metadata"]["feature_cols"]
    input_row = build_input_row(data, **inputs)

    pred_label = str(clf.predict(input_row[feature_cols])[0])
    try:
        proba_raw = clf.predict_proba(input_row[feature_cols])[0]
        proba_series = pd.Series(proba_raw, index=list(clf.classes_)).reindex(LABEL_ORDER).fillna(0).values
    except Exception:
        proba_series = np.array([0.0, 0.0, 0.0])
        proba_series[LABEL_TO_NUM.get(pred_label, 1)] = 1.0
    pred_pcf = float(max(reg.predict(input_row[feature_cols])[0], 0))

    industry_group = inputs["industry_group"]
    industry_data = data.loc[data["industry_group"].eq(industry_group), TARGET_COL].dropna().astype(float)
    industry_median = float(industry_data.median()) if len(industry_data) else float(data[TARGET_COL].median())
    global_median = float(data[TARGET_COL].median())
    diff_pct = (pred_pcf - industry_median) / max(industry_median, 1e-9) * 100
    bench_text, industry_pct, global_pct, n_ref = build_benchmark_analysis(data, pred_pcf, industry_group, industry_median, global_median)

    kpi_class = "kpi-green" if pred_label == "Low" else "kpi-orange" if pred_label == "Medium" else "kpi-red"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi-card {kpi_class}'><div class='kpi-title'>Phân loại carbon</div><div class='kpi-value'>{LABEL_VI.get(pred_label, pred_label)}</div><div class='kpi-note'>Low / Medium / High được tạo bằng ngưỡng train-only</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>PCF ước lượng</div><div class='kpi-value'>{fmt_num(pred_pcf)}</div><div class='kpi-note'>kg CO₂e / functional unit</div></div>", unsafe_allow_html=True)
    with c3:
        sign = "thấp hơn" if diff_pct < 0 else "cao hơn"
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>So với trung vị ngành</div><div class='kpi-value'>{abs(diff_pct):.1f}%</div><div class='kpi-note'>Sản phẩm {sign} mức trung vị ngành</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>Diễn giải nhanh</div>", unsafe_allow_html=True)
    if pred_label == "Low":
        quick = "Sản phẩm đang được xếp vào nhóm phát thải thấp. Đây là tín hiệu tích cực khi so với dữ liệu tham chiếu, nhưng không thay thế chứng nhận LCA chính thức."
    elif pred_label == "Medium":
        quick = "Sản phẩm nằm ở vùng phát thải trung bình. Nên tiếp tục theo dõi benchmark ngành và xem yếu tố nào có thể tối ưu để giảm PCF."
    else:
        quick = "Sản phẩm thuộc nhóm phát thải cao. Cần ưu tiên rà soát khối lượng, vật liệu, năng lượng và tỷ trọng phát thải vòng đời."
    st.markdown(f"<div class='success-box'>{quick}</div></div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><div class='section-title'>Xác suất phân loại</div><div class='card-subtitle'>Mức tin cậy tương đối của mô hình với từng nhãn phát thải.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_probability_bar(proba_series), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='section-title'>Tỷ trọng vòng đời</div><div class='card-subtitle'>Cơ cấu Upstream / Operations / Downstream sau khi chuẩn hóa.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_lifecycle_donut(inputs["upstream_frac"], inputs["operations_frac"], inputs["downstream_frac"]), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>So sánh PCF với ngành</div><div class='card-subtitle'>Đặt sản phẩm trong bối cảnh cùng nhóm ngành để đánh giá mức phát thải tương đối.</div>", unsafe_allow_html=True)
    bc1, bc2 = st.columns([1.15, 1])
    with bc1:
        st.plotly_chart(plot_benchmark(pred_pcf, industry_median, global_median), use_container_width=True, config=PLOTLY_CONFIG)
    with bc2:
        ratio = pred_pcf / max(industry_median, 1e-9)
        abs_gap = pred_pcf - industry_median
        st.markdown(f"""
        <div class='insight-grid'>
          <div class='insight-card'><div class='insight-label'>Vị trí trong nhóm ngành</div><div class='insight-value'>{industry_pct:.1f}%</div><div class='insight-note'>Tỷ lệ mẫu cùng ngành có PCF thấp hơn hoặc bằng sản phẩm này.</div></div>
          <div class='insight-card'><div class='insight-label'>Tỷ lệ so với trung vị ngành</div><div class='insight-value'>{ratio:.2f}x</div><div class='insight-note'>Nhỏ hơn 1 là thấp hơn benchmark; lớn hơn 1 là cao hơn benchmark.</div></div>
          <div class='insight-card'><div class='insight-label'>Số mẫu tham chiếu</div><div class='insight-value'>{n_ref}</div><div class='insight-note'>Số mẫu cùng nhóm ngành được dùng để so sánh.</div></div>
          <div class='insight-card'><div class='insight-label'>Chênh lệch tuyệt đối</div><div class='insight-value'>{fmt_num(abs_gap)}</div><div class='insight-note'>kg CO₂e so với trung vị ngành.</div></div>
        </div>
        <div class='success-box'>{bench_text}</div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>Giải thích yếu tố ảnh hưởng</div><div class='card-subtitle'>Giải thích cục bộ cho riêng sản phẩm đang được dự báo. Biểu đồ chỉ hiển thị top 6 yếu tố để tránh chồng chữ.</div>", unsafe_allow_html=True)
    exp = get_local_factor_effect(package, input_row, data)
    if exp.empty:
        st.info("Chưa có dữ liệu giải thích yếu tố ảnh hưởng.")
    else:
        top_names = ", ".join(exp.head(3)["feature_vi"].tolist())
        st.markdown(f"<div class='explain-note'>Các yếu tố ảnh hưởng mạnh nhất tới dự báo hiện tại gồm: {top_names}. Cột xanh làm tăng xu hướng mô hình xếp sản phẩm vào nhãn hiện tại, còn cột cam kéo dự báo theo chiều ngược lại. Đây là quan hệ dự đoán của mô hình, không được hiểu là quan hệ nhân quả tuyệt đối.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_factor_effect(exp), use_container_width=True, config=PLOTLY_CONFIG)
        table = exp.head(6).copy()
        table["Tên yếu tố"] = table["feature_vi"]
        table["Mức tác động"] = table["impact"].map(lambda x: f"{x:+.4f}")
        table["Chiều tác động"] = np.where(table["impact"] >= 0, "Tăng xu hướng nhãn dự báo", "Giảm xu hướng nhãn dự báo")
        st.markdown("<p class='small-muted'>Bảng chi tiết dùng để xem đầy đủ tên yếu tố nếu nhãn trên biểu đồ đã rút gọn.</p>", unsafe_allow_html=True)
        st.dataframe(table[["Tên yếu tố", "Mức tác động", "Chiều tác động"]], hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📦 Dự báo hàng loạt bằng CSV"):
        up = st.file_uploader("Upload CSV có cấu trúc gần giống Carbon Catalogue", type=["csv"])
        if up is not None:
            batch = pd.read_csv(up)
            from carbon_utils import add_model_features
            batch = add_model_features(batch)
            for c in feature_cols:
                if c not in batch.columns:
                    batch[c] = np.nan
            pred_labels = clf.predict(batch[feature_cols])
            pred_pcf = np.maximum(reg.predict(batch[feature_cols]), 0)
            out = batch.copy()
            out["predicted_carbon_class"] = pred_labels
            out["predicted_pcf_kg_co2e"] = pred_pcf
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button("Tải kết quả CSV", out.to_csv(index=False, encoding="utf-8-sig"), file_name="ecopredict_batch_predictions.csv", mime="text/csv")


def data_page(data: pd.DataFrame) -> None:
    render_hero()
    st.markdown("<div class='card'><div class='section-title'>Tổng quan dữ liệu Carbon Catalogue</div><div class='card-subtitle'>Dữ liệu sau chuẩn hóa, lọc PCF và khối lượng hợp lệ.</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số mẫu", f"{len(data):,}")
    c2.metric("Số cột", f"{data.shape[1]:,}")
    c3.metric("Trung vị PCF", fmt_num(float(data[TARGET_COL].median())))
    c4.metric("PCF cao nhất", fmt_num(float(data[TARGET_COL].max())))
    st.markdown("</div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><div class='section-title'>Phân phối PCF</div>", unsafe_allow_html=True)
        hist = go.Figure(go.Histogram(x=data[TARGET_COL], nbinsx=45, marker_color="#047857", showlegend=False, name=""))
        hist.update_layout(height=350, xaxis_title="PCF kg CO₂e", yaxis_title="Số mẫu", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), showlegend=False)
        st.plotly_chart(hist, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='section-title'>Top nhóm ngành theo số mẫu</div>", unsafe_allow_html=True)
        vc = data["industry_group"].value_counts().head(10).sort_values()
        fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", marker_color="#10b981", showlegend=False, name=""))
        fig.update_layout(height=350, margin=dict(l=160, r=20, t=10, b=30), xaxis_title="Số mẫu", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>Bảng dữ liệu đã xử lý</div>", unsafe_allow_html=True)
    st.dataframe(data.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def model_page(package: dict[str, Any]) -> None:
    render_hero()
    st.markdown("<div class='card'><div class='section-title'>Thông tin nâng cao về mô hình</div><div class='card-subtitle'>Phần này dành cho giảng viên hoặc người muốn xem chi tiết kỹ thuật.</div>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>Best classifier: {package.get('classifier_name','')}</span><span class='badge'>Best regressor: {package.get('regressor_name','')}</span><span class='badge'>Train/Test: {package['metadata'].get('split','')}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if "classification_summary" in package:
        st.markdown("<div class='card'><div class='section-title'>So sánh mô hình phân loại</div>", unsafe_allow_html=True)
        st.dataframe(package["classification_summary"], use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    if "regression_summary" in package:
        st.markdown("<div class='card'><div class='section-title'>So sánh mô hình hồi quy</div>", unsafe_allow_html=True)
        st.dataframe(package["regression_summary"], use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    fig_dir = Path("outputs/figures")
    imgs = [
        ("Confusion Matrix", fig_dir / "confusion_matrix_best_classifier.png"),
        ("ROC Curves", fig_dir / "roc_curves_comparison.png"),
        ("Calibration Curve", fig_dir / "calibration_curve_high_class.png"),
        ("Permutation Importance", fig_dir / "permutation_importance_best_classifier.png"),
        ("Predicted vs Actual", fig_dir / "regression_predicted_vs_actual.png"),
        ("Residual Distribution", fig_dir / "regression_residual_distribution.png"),
    ]
    for title, path in imgs:
        if path.exists():
            st.markdown(f"<div class='card'><div class='section-title'>{title}</div>", unsafe_allow_html=True)
            st.image(str(path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


def guide_page() -> None:
    render_hero()
    st.markdown("""
    <div class='card'>
      <div class='section-title'>Hướng dẫn sử dụng</div>
      <div class='success-box'>
      1. Vào trang <b>Dự báo</b>, nhập thông tin sản phẩm ở sidebar.<br>
      2. Xem PCF ước lượng, nhãn phát thải, so sánh với ngành và yếu tố ảnh hưởng.<br>
      3. Vào <b>Dữ liệu</b> để xem tổng quan Carbon Catalogue.<br>
      4. Vào <b>Thông tin nâng cao</b> để xem metric, ROC, calibration, permutation importance và hồi quy.
      </div>
    </div>
    """, unsafe_allow_html=True)


def bootstrap() -> tuple[pd.DataFrame, dict[str, Any] | None]:
    data = load_data_cached(str(DATA_PATH))
    return data, load_model_cached()
    data = load_data_cached(str(DATA_PATH))
    if not MODEL_PATH.exists():
        st.warning("Chưa có model package. Hãy chạy `python train_advanced_models.py` trước, hoặc bấm nút bên dưới để huấn luyện ngay.")
        if st.button("Huấn luyện model ngay"):
            with st.spinner("Đang huấn luyện mô hình, vui lòng chờ..."):
                from train_advanced_models import main as train_main
                train_main(DATA_PATH)
            st.cache_resource.clear()
            st.rerun()
        return data, None
    return data, load_model_cached(str(MODEL_PATH))


def main() -> None:
    data, package = bootstrap()
    if data is None:
        return
    with st.sidebar:
        page = st.radio("Điều hướng", ["Dự báo", "Dữ liệu", "Thông tin nâng cao", "Hướng dẫn"], label_visibility="collapsed")
    if package is None:
        data_page(data)
        return
    if page == "Dự báo":
        prediction_page(data, package)
    elif page == "Dữ liệu":
        data_page(data)
    elif page == "Thông tin nâng cao":
        model_page(package)
    else:
        guide_page()


if __name__ == "__main__":
    main()
