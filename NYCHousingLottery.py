"""
NYC Housing Lottery Finder
A Streamlit app to browse, map, and analyze NYC affordable housing lotteries

Key improvement:
- Enrich + normalize the dataframe ONCE (optional overrides merge) and use the same enriched dataframe
  across List View + Unit Distribution (and all tabs) so counts stay consistent.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import folium
from streamlit_folium import st_folium
import plotly.express as px
from typing import Optional, List, Dict, Tuple

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="NYC Housing Lottery Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# Column definitions (friendly labels + descriptions)
# ---------------------------
COLUMN_DEFINITIONS = {
    "lottery_id": {"display_name": "Lottery ID", "description": "Unique identifier for the housing lottery"},
    "lottery_name": {"display_name": "Lottery Name", "description": "Name of the housing development/lottery"},
    "lottery_status": {"display_name": "Lottery Status", "description": "Current status: Open, Closed, or All Units Filled"},
    "development_type": {"display_name": "Development Type", "description": "Type of housing: Rental or Homeownership"},
    "lottery_start_date": {"display_name": "Start Date", "description": "Date when lottery applications open"},
    "lottery_end_date": {"display_name": "End Date", "description": "Application deadline for the lottery"},
    "building_count": {"display_name": "Building Count", "description": "Number of buildings in the development"},
    "unit_count": {"display_name": "Total Units", "description": "Total number of available housing units"},
    "unit_distribution_studio": {"display_name": "Studio Units", "description": "Number of studio apartments available"},
    "unit_distribution_1_bedroom": {"display_name": "1-Bedroom Units", "description": "Number of 1-bedroom apartments available"},
    "unit_distribution_2_bedrooms": {"display_name": "2-Bedroom Units", "description": "Number of 2-bedroom apartments available"},
    "unit_distribution_3_bedrooms": {"display_name": "3-Bedroom Units", "description": "Number of 3-bedroom apartments available"},
    "unit_distribution_4_bedroom": {"display_name": "4+ Bedroom Units", "description": "Number of 4 or more bedroom apartments available"},
    "applied_income_ami_category_extremely_low_income": {"display_name": "Extremely Low Income", "description": "Units for households at 0-30% AMI"},
    "applied_income_ami_category_very_low_income": {"display_name": "Very Low Income", "description": "Units for households at 31-50% AMI"},
    "applied_income_ami_category_low_income": {"display_name": "Low Income", "description": "Units for households at 51-80% AMI"},
    "applied_income_ami_category_moderate_income": {"display_name": "Moderate Income", "description": "Units for households at 81-120% AMI"},
    "applied_income_ami_category_middle_income": {"display_name": "Middle Income", "description": "Units for households at 121-165% AMI"},
    "applied_income_ami_category_above_middle_income": {"display_name": "Above Middle Income", "description": "Units for households above 165% AMI"},
    "lottery_mobility_percentage": {"display_name": "Mobility %", "description": "Percentage of units for applicants with mobility disabilities"},
    "lottery_vision_hearing_percentage": {"display_name": "Vision/Hearing %", "description": "Percentage of units for applicants with vision/hearing disabilities"},
    "lottery_community_board_percentage": {"display_name": "Community Board %", "description": "Percentage of units reserved for community board residents"},
    "lottery_municipal_employee_military_veteran_percentage": {"display_name": "Municipal/Veteran %", "description": "Percentage for municipal employees and military veterans"},
    "lottery_nycha_percentage": {"display_name": "NYCHA %", "description": "Percentage of units for NYCHA residents"},
    "lottery_senior_percentage": {"display_name": "Senior %", "description": "Percentage of units reserved for seniors"},
    "borough": {"display_name": "Borough", "description": "NYC borough where the development is located"},
    "postcode": {"display_name": "Zip Code", "description": "Postal code of the development"},
    "community_board": {"display_name": "Community Board", "description": "NYC Community Board district number"},
    "latitude": {"display_name": "Latitude", "description": "Geographic latitude coordinate"},
    "longitude": {"display_name": "Longitude", "description": "Geographic longitude coordinate"},
}

UNIT_DIST_COLS = [
    "unit_distribution_studio",
    "unit_distribution_1_bedroom",
    "unit_distribution_2_bedrooms",
    "unit_distribution_3_bedrooms",
    "unit_distribution_4_bedroom",
]

AMI_COLS = [
    "applied_income_ami_category_extremely_low_income",
    "applied_income_ami_category_very_low_income",
    "applied_income_ami_category_low_income",
    "applied_income_ami_category_moderate_income",
    "applied_income_ami_category_middle_income",
    "applied_income_ami_category_above_middle_income",
]

LOTTERY_PCT_COLS = [
    "lottery_mobility_percentage",
    "lottery_vision_hearing_percentage",
    "lottery_community_board_percentage",
    "lottery_municipal_employee_military_veteran_percentage",
    "lottery_nycha_percentage",
    "lottery_senior_percentage",
]

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown(
    """
<style>
  @media (max-width: 768px) {
    .block-container { padding: 1rem 0.5rem; }
    .stSelectbox, .stDateInput { min-width: 100%; }
  }
  .lottery-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.8rem 0;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  }
  .lottery-card h3 { margin: 0 0 0.5rem 0; font-size: 1.1rem; color: white; }
  .lottery-card p { margin: 0.3rem 0; font-size: 0.9rem; opacity: 0.95; }
  .status-open  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
  .status-closed{ background: linear-gradient(135deg, #636363 0%, #a2ab58 100%); }
  .status-filled{ background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
  .main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 10px;
    margin-bottom: 1.5rem;
  }
  .main-header h1 { color: #e94560; font-size: 2rem; margin: 0; }
  .main-header p  { color: #eaeaea; margin: 0.5rem 0 0 0; }
  .stat-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border-left: 4px solid #667eea;
  }
  .stat-number { font-size: 1.8rem; font-weight: bold; color: #667eea; }
  .stat-label  { font-size: 0.85rem; color: #666; }
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
def safe_numeric_sum(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.fillna(0).sum())

def safe_numeric_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return float("nan")
    return float(s.mean())

def detect_related_columns(df: pd.DataFrame, keywords: List[str]) -> List[str]:
    kws = [k.lower() for k in keywords]
    hits = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in kws):
            hits.append(c)
    return sorted(hits)

def _safe_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype="object")
    return df[col].astype(str)

def _to_timestamp(d: Optional[date | datetime]) -> Optional[pd.Timestamp]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return pd.Timestamp(d.date())
    return pd.Timestamp(d)

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# Data fetch
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_lottery_data() -> pd.DataFrame:
    api_url = "https://data.cityofnewyork.us/resource/vy5i-a666.json"
    params = {"$limit": 5000, "$order": "lottery_end_date DESC"}
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)

        for col in ["lottery_start_date", "lottery_end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        numeric_candidates = [
            "latitude", "longitude", "unit_count", "building_count",
            *UNIT_DIST_COLS, *AMI_COLS, *LOTTERY_PCT_COLS,
        ]
        for col in numeric_candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# ---------------------------
# Enrich + Normalize (once)
# ---------------------------
def _normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _ensure_unit_columns_exist(df: pd.DataFrame) -> pd.DataFrame:
    for col in UNIT_DIST_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def _coerce_unit_cols_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in UNIT_DIST_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _standardize_overrides_columns(overrides: pd.DataFrame) -> pd.DataFrame:
    o = overrides.copy()
    o.columns = [c.strip() for c in o.columns]

    id_candidates = ["lottery_id", "id", "lotteryid", "Lottery ID", "LOTTERY_ID"]
    id_col = next((c for c in id_candidates if c in o.columns), None)
    if id_col is None:
        raise ValueError("Overrides file must contain a lottery_id column (e.g., 'lottery_id').")

    rename_map = {id_col: "lottery_id"}

    size_candidates = {
        "studio": ["studio", "studios", "studio_units", "unit_distribution_studio"],
        "1br": ["1br", "onebr", "one_br", "1_bedroom", "unit_distribution_1_bedroom", "unit_distribution_1_bedrooms"],
        "2br": ["2br", "twobr", "two_br", "2_bedroom", "unit_distribution_2_bedrooms", "unit_distribution_2_bedroom"],
        "3br": ["3br", "threebr", "three_br", "3_bedroom", "unit_distribution_3_bedrooms", "unit_distribution_3_bedroom"],
        "4br": ["4br", "fourbr", "four_br", "4_bedroom", "4plus", "4_plus", "unit_distribution_4_bedroom", "unit_distribution_4_bedrooms"],
    }
    for target, cands in size_candidates.items():
        found = next((c for c in cands if c in o.columns), None)
        if found is not None:
            rename_map[found] = target

    o = o.rename(columns=rename_map)
    keep = ["lottery_id", "studio", "1br", "2br", "3br", "4br"]
    for k in keep:
        if k not in o.columns:
            o[k] = pd.NA

    o["lottery_id"] = _normalize_id_series(o["lottery_id"])
    for k in ["studio", "1br", "2br", "3br", "4br"]:
        o[k] = pd.to_numeric(o[k], errors="coerce")

    return o[keep]

def enrich_and_normalize(df: pd.DataFrame, overrides: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "lottery_id" in out.columns:
        out["lottery_id"] = _normalize_id_series(out["lottery_id"])
    else:
        out["lottery_id"] = pd.NA

    out = _ensure_unit_columns_exist(out)
    out = _coerce_unit_cols_numeric(out)

    # ------------------------------------------------------------------
    # 1) Normalize *alternate* NYC Open Data schemas into the canonical
    #    unit_distribution_* columns used everywhere in the app.
    #
    # Why: the API sometimes returns columns like `unit_distribution_1bed`
    # instead of `unit_distribution_1_bedroom`. If we don't fold these
    # alternates in, charts/tabs that rely on canonical columns will show
    # zeros (or only Studio), even though the data is present.
    # ------------------------------------------------------------------

    UNIT_ALT_MAP: Dict[str, List[str]] = {
        "unit_distribution_studio": [
            "unit_distribution_studio",
            "unit_distribution_studios",
            "studio_units",
            "studios",
            "unit_distribution_stu",
        ],
        "unit_distribution_1_bedroom": [
            "unit_distribution_1_bedroom",
            "unit_distribution_1_bedrooms",
            "unit_distribution_1bed",
            "unit_distribution_1_bed",
            "unit_distribution_1br",
            "one_bedroom_units",
            "1_bedroom_units",
        ],
        "unit_distribution_2_bedrooms": [
            "unit_distribution_2_bedrooms",
            "unit_distribution_2_bedroom",
            "unit_distribution_2bed",
            "unit_distribution_2_bed",
            "unit_distribution_2br",
            "two_bedroom_units",
            "2_bedroom_units",
        ],
        "unit_distribution_3_bedrooms": [
            "unit_distribution_3_bedrooms",
            "unit_distribution_3_bedroom",
            "unit_distribution_3bed",
            "unit_distribution_3_bed",
            "unit_distribution_3br",
            "three_bedroom_units",
            "3_bedroom_units",
        ],
        "unit_distribution_4_bedroom": [
            "unit_distribution_4_bedroom",
            "unit_distribution_4_bedrooms",
            "unit_distribution_4bed",
            "unit_distribution_4_bed",
            "unit_distribution_4br",
            "unit_distribution_4_plus",
            "four_bedroom_units",
            "4_bedroom_units",
        ],
    }

    # Build lowercase lookup -> actual column name for robust matching
    col_lookup = {c.lower(): c for c in out.columns}

    for canonical, candidates in UNIT_ALT_MAP.items():
        canonical_actual = col_lookup.get(canonical.lower())
        if not canonical_actual:
            # ensure canonical exists (should already), but be safe
            out[canonical] = 0
            canonical_actual = canonical

        # Find the first alternate candidate that exists (excluding canonical itself)
        alt_actual: Optional[str] = None
        for cand in candidates:
            if cand.lower() == canonical.lower():
                continue
            if cand.lower() in col_lookup:
                alt_actual = col_lookup[cand.lower()]
                break

        if not alt_actual:
            continue

        # Coerce alt numeric
        alt_vals = pd.to_numeric(out[alt_actual], errors="coerce")

        # Fill canonical values when missing OR equal to 0 while alt is > 0
        can_vals = pd.to_numeric(out[canonical_actual], errors="coerce")
        mask = (can_vals.isna() | (can_vals.fillna(0) == 0)) & (alt_vals.fillna(0) > 0)
        if mask.any():
            out.loc[mask, canonical_actual] = alt_vals.loc[mask]

    # Re-coerce canonical unit columns (now that we may have copied values)
    out = _coerce_unit_cols_numeric(out)

    if overrides is not None and not overrides.empty:
        o = _standardize_overrides_columns(overrides)
        out = out.merge(o, on="lottery_id", how="left")

        pairs = {
            "unit_distribution_studio": "studio",
            "unit_distribution_1_bedroom": "1br",
            "unit_distribution_2_bedrooms": "2br",
            "unit_distribution_3_bedrooms": "3br",
            "unit_distribution_4_bedroom": "4br",
        }
        for primary_col, ovr_col in pairs.items():
            p = pd.to_numeric(out[primary_col], errors="coerce")
            s = pd.to_numeric(out[ovr_col], errors="coerce")
            use_override = (p.isna() | (p <= 0)) & (s.notna() & (s > 0))
            out.loc[use_override, primary_col] = s.loc[use_override]

        out = out.drop(columns=["studio", "1br", "2br", "3br", "4br"], errors="ignore")

    for col in UNIT_DIST_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["unit_breakdown_present"] = (
        out[["unit_distribution_1_bedroom", "unit_distribution_2_bedrooms", "unit_distribution_3_bedrooms", "unit_distribution_4_bedroom"]]
        .sum(axis=1)
        .fillna(0) > 0
    )
    return out

# ---------------------------
# Filtering
# ---------------------------
def filter_data(
    df: pd.DataFrame,
    borough: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[date | datetime] = None,
    end_date: Optional[date | datetime] = None,
    development_type: Optional[str] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    filtered = df.copy()

    if borough and borough != "All Boroughs" and "borough" in filtered.columns:
        b = filtered["borough"].astype(str)
        filtered = filtered[b.str.upper() == borough.upper()]

    if status and status != "All Statuses" and "lottery_status" in filtered.columns:
        s = filtered["lottery_status"].astype(str)
        filtered = filtered[s.str.contains(status, case=False, na=False)]

    if development_type and development_type != "All Types" and "development_type" in filtered.columns:
        d = filtered["development_type"].astype(str)
        filtered = filtered[d.str.contains(development_type, case=False, na=False)]

    sd = _to_timestamp(start_date)
    ed = _to_timestamp(end_date)

    if sd is not None and "lottery_end_date" in filtered.columns:
        filtered = filtered[filtered["lottery_end_date"] >= sd]

    if ed is not None and "lottery_start_date" in filtered.columns:
        filtered = filtered[filtered["lottery_start_date"] <= ed]

    return filtered

# ---------------------------
# UI helpers
# ---------------------------
def get_status_class(status: str) -> str:
    status = (status or "").lower()
    if "open" in status:
        return "status-open"
    if "filled" in status:
        return "status-filled"
    return "status-closed"

def create_map(df: pd.DataFrame) -> folium.Map:
    center_lat, center_lon = 40.7128, -74.0060
    if df is None or df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    map_df = df.dropna(subset=["latitude", "longitude"])
    if not map_df.empty:
        center_lat = float(pd.to_numeric(map_df["latitude"], errors="coerce").dropna().mean())
        center_lon = float(pd.to_numeric(map_df["longitude"], errors="coerce").dropna().mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")
    for _, row in map_df.iterrows():
        status = str(row.get("lottery_status", "")).lower()
        color = "green" if "open" in status else ("red" if "filled" in status else "blue")

        start_dt = row.get("lottery_start_date", pd.NaT)
        end_dt = row.get("lottery_end_date", pd.NaT)

        popup_html = f"""
        <div style="width: 260px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{row.get('lottery_name', 'N/A')}</h4>
            <p style="margin: 5px 0;"><strong>Status:</strong> {row.get('lottery_status', 'N/A')}</p>
            <p style="margin: 5px 0;"><strong>Borough:</strong> {row.get('borough', 'N/A')}</p>
            <p style="margin: 5px 0;"><strong>Units:</strong> {row.get('unit_count', 'N/A')}</p>
            <p style="margin: 5px 0;"><strong>Type:</strong> {row.get('development_type', 'N/A')}</p>
            <p style="margin: 5px 0;"><strong>Start:</strong> {pd.to_datetime(start_dt).strftime('%m/%d/%Y') if pd.notna(start_dt) else 'N/A'}</p>
            <p style="margin: 5px 0;"><strong>End:</strong> {pd.to_datetime(end_dt).strftime('%m/%d/%Y') if pd.notna(end_dt) else 'N/A'}</p>
        </div>
        """

        lat = pd.to_numeric(row.get("latitude", None), errors="coerce")
        lon = pd.to_numeric(row.get("longitude", None), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue

        folium.Marker(
            location=[float(lat), float(lon)],
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color=color, icon="home", prefix="fa"),
            tooltip=row.get("lottery_name", "Housing Lottery"),
        ).add_to(m)

    return m

def display_lottery_card(row: pd.Series) -> None:
    status_class = get_status_class(row.get("lottery_status", ""))
    start_date = pd.to_datetime(row.get("lottery_start_date", pd.NaT), errors="coerce")
    end_date = pd.to_datetime(row.get("lottery_end_date", pd.NaT), errors="coerce")
    start_str = start_date.strftime("%b %d, %Y") if pd.notna(start_date) else "TBD"
    end_str = end_date.strftime("%b %d, %Y") if pd.notna(end_date) else "TBD"

    st.markdown(
        f"""
    <div class="lottery-card {status_class}">
        <h3>🏠 {row.get('lottery_name', 'N/A')}</h3>
        <p><strong>📍 Borough:</strong> {row.get('borough', 'N/A')}</p>
        <p><strong>📋 Status:</strong> {row.get('lottery_status', 'N/A')}</p>
        <p><strong>🏗️ Type:</strong> {row.get('development_type', 'N/A')}</p>
        <p><strong>🏢 Units:</strong> {row.get('unit_count', 'N/A')} units in {row.get('building_count', 'N/A')} building(s)</p>
        <p><strong>📅 Application Period:</strong> {start_str} - {end_str}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

def display_detailed_lottery_info(row: pd.Series) -> None:
    with st.expander(f"🏠 {row.get('lottery_name', 'N/A')} - {row.get('lottery_status', 'N/A')}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Basic Information")
            for key in ["lottery_id", "lottery_status", "development_type", "lottery_start_date", "lottery_end_date", "unit_count"]:
                if key in row.index:
                    v = row.get(key, "N/A")
                    if pd.isna(v):
                        v = "N/A"
                    elif "date" in key and pd.notna(v):
                        v = pd.to_datetime(v).strftime("%m/%d/%Y")
                    st.markdown(f"**{COLUMN_DEFINITIONS.get(key, {}).get('display_name', key)}:** {v}")

        with col2:
            st.markdown("#### Unit Distribution")
            unit_map = [
                ("Studio", "unit_distribution_studio"),
                ("1BR", "unit_distribution_1_bedroom"),
                ("2BR", "unit_distribution_2_bedrooms"),
                ("3BR", "unit_distribution_3_bedrooms"),
                ("4+BR", "unit_distribution_4_bedroom"),
            ]
            for label, col in unit_map:
                val = row.get(col, 0)
                val = 0 if pd.isna(val) else val
                st.markdown(f"**{label}:** {int(float(val))}")

# ---------------------------
# Tabs
# ---------------------------
def render_unit_distribution_tab(df: pd.DataFrame) -> None:
    st.markdown("### 🏢 Unit Distribution Analysis")
    st.markdown("Analyze the distribution of unit sizes across housing lotteries (uses enriched dataframe).")

    st.markdown("#### Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        boroughs = ["All Boroughs"] + (sorted(df["borough"].dropna().unique().tolist()) if "borough" in df.columns else [])
        ud_borough = st.selectbox("Borough", boroughs, key="ud_borough")
    with c2:
        ud_status = st.selectbox("Status", ["All Statuses", "Open", "Closed", "Filled"], key="ud_status")
    with c3:
        types = ["All Types"] + (sorted(df["development_type"].dropna().unique().tolist()) if "development_type" in df.columns else [])
        ud_type = st.selectbox("Development Type", types, key="ud_type")

    filtered = filter_data(df, borough=ud_borough, status=ud_status, development_type=ud_type)
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    all_labels = ["Studio", "1BR", "2BR", "3BR", "4+BR"]
    selected_labels = st.multiselect("Unit sizes to include", all_labels, default=all_labels, key="ud_unit_sizes")

    chart_mode = st.radio("Chart view", ["Both", "Pie", "Bar"], horizontal=True, key="ud_chart_mode")
    include_zeros_in_charts = st.checkbox("Include zero categories in charts", value=False, key="ud_include_zeros_in_charts")

    col_map = {
        "Studio": "unit_distribution_studio",
        "1BR": "unit_distribution_1_bedroom",
        "2BR": "unit_distribution_2_bedrooms",
        "3BR": "unit_distribution_3_bedrooms",
        "4+BR": "unit_distribution_4_bedroom",
    }

    totals = {lbl: safe_numeric_sum(filtered[col_map[lbl]]) for lbl in selected_labels}
    summary_df = pd.DataFrame({"Unit Size": list(totals.keys()), "Units": list(totals.values())})
    order = {"Studio": 0, "1BR": 1, "2BR": 2, "3BR": 3, "4+BR": 4}
    summary_df["__o"] = summary_df["Unit Size"].map(lambda x: order.get(x, 99))
    summary_df = summary_df.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    total_units_all = float(summary_df["Units"].sum())
    summary_df["Share"] = (summary_df["Units"] / total_units_all).fillna(0).map(lambda x: f"{x:.1%}") if total_units_all > 0 else "0.0%"

    st.markdown("#### Summary")
    st.dataframe(summary_df, width="stretch", height=230)

    chart_df = summary_df.copy()
    if not include_zeros_in_charts:
        chart_df = chart_df[chart_df["Units"] > 0].copy()

    if not chart_df.empty:
        left, right = st.columns(2)
        if chart_mode in ("Both", "Pie"):
            with left:
                fig_pie = px.pie(chart_df, values="Units", names="Unit Size", title="Unit Distribution by Size", hole=0.35)
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, width="stretch")
        if chart_mode in ("Both", "Bar"):
            with (right if chart_mode == "Both" else left):
                fig_bar = px.bar(chart_df, x="Unit Size", y="Units", title="Total Units by Size", labels={"Units": "Number of Units"})
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, width="stretch")
    else:
        st.info("All selected unit sizes are 0 for these filters.")

    st.markdown("#### Detailed Data")
    base_cols = [c for c in ["lottery_id", "lottery_name", "borough", "lottery_status", "development_type", "unit_count"] if c in filtered.columns]
    display_df = filtered[base_cols].copy()
    for lbl in selected_labels:
        display_df[f"{lbl} Units"] = pd.to_numeric(filtered[col_map[lbl]], errors="coerce").fillna(0).astype(int)

    rename_map = {"lottery_id": "Lottery ID", "lottery_name": "Lottery Name", "borough": "Borough", "lottery_status": "Status", "development_type": "Development Type", "unit_count": "Total Units"}
    display_df = display_df.rename(columns=rename_map)
    st.dataframe(display_df, width="stretch", height=350)

    st.download_button(
        "📥 Download Unit Distribution Data (CSV)",
        data=convert_df_to_csv(display_df),
        file_name=f"unit_distribution_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_unit_dist",
    )

    with st.expander("🔎 Diagnostics (optional)"):
        st.write("Detected unit-related columns:", detect_related_columns(filtered, ["unit", "bed", "studio"]))
        st.write("Rows with non-studio breakdown present:", int(filtered["unit_breakdown_present"].sum()) if "unit_breakdown_present" in filtered.columns else "n/a")

# ---------------------------
# Main app
# ---------------------------
def main() -> None:
    st.markdown(
        """
    <div class="main-header">
        <h1>🏠 NYC Housing Lottery Finder</h1>
        <p>Find and explore affordable housing opportunities across New York City</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar: optional secondary source overrides
    with st.sidebar:
        st.markdown("## 🔧 Data Enrichment")
        st.caption("Optional: upload a CSV with bedroom breakdowns to enrich missing unit distributions.")
        st.caption("Required column: lottery_id. Optional columns: studio, 1br, 2br, 3br, 4br (or similar).")
        overrides_file = st.file_uploader("Upload overrides CSV", type=["csv"], key="overrides_csv")
        overrides_df = None
        if overrides_file is not None:
            try:
                overrides_df = pd.read_csv(overrides_file)
                st.success(f"Loaded overrides: {len(overrides_df):,} rows")
            except Exception as e:
                st.error(f"Could not read overrides CSV: {e}")
                overrides_df = None

        show_diag = st.checkbox("Show enrichment diagnostics", value=False, key="show_enrich_diag")

    with st.spinner("Loading lottery data..."):
        raw_df = fetch_lottery_data()

    if raw_df.empty:
        st.error("Unable to load lottery data. Please try again later.")
        return

    # ✅ Enrich + normalize ONCE, then use everywhere
    try:
        df = enrich_and_normalize(raw_df, overrides_df)
    except Exception as e:
        st.error(f"Enrichment error: {e}")
        df = enrich_and_normalize(raw_df, None)

    if show_diag:
        with st.expander("🧪 Enrichment Diagnostics"):
            st.write("Raw unit-like columns:", detect_related_columns(raw_df, ["unit_distribution", "bed", "studio"]))
            st.write("Rows with non-studio breakdown present:", int(df["unit_breakdown_present"].sum()) if "unit_breakdown_present" in df.columns else "n/a")
            st.write(df[["lottery_id", "unit_count", *UNIT_DIST_COLS]].head(25))

    st.markdown("### 🔍 Filter Lotteries")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        boroughs = ["All Boroughs"] + (sorted(df["borough"].dropna().unique().tolist()) if "borough" in df.columns else [])
        selected_borough = st.selectbox("Borough", boroughs, key="main_borough_filter")

    with col2:
        selected_status = st.selectbox("Status", ["All Statuses", "Open", "Closed", "Filled"], key="main_status_filter")

    with col3:
        start_date = st.date_input("From Date", value=(datetime.now() - timedelta(days=30)).date(), key="main_start_date")

    with col4:
        end_date = st.date_input("To Date", value=(datetime.now() + timedelta(days=180)).date(), key="main_end_date")

    filtered_df = filter_data(df, borough=selected_borough, status=selected_status, start_date=start_date, end_date=end_date)

    st.markdown("### 📊 Summary Statistics")
    sc1, sc2, sc3, sc4 = st.columns(4)

    with sc1:
        st.markdown(f"""<div class="stat-card"><div class="stat-number">{len(filtered_df)}</div><div class="stat-label">Total Lotteries</div></div>""", unsafe_allow_html=True)

    with sc2:
        open_count = int(_safe_str_series(filtered_df, "lottery_status").str.contains("Open", case=False, na=False).sum()) if "lottery_status" in filtered_df.columns else 0
        st.markdown(f"""<div class="stat-card"><div class="stat-number">{open_count}</div><div class="stat-label">Currently Open</div></div>""", unsafe_allow_html=True)

    with sc3:
        total_units = int(pd.to_numeric(filtered_df["unit_count"], errors="coerce").fillna(0).sum()) if "unit_count" in filtered_df.columns else 0
        st.markdown(f"""<div class="stat-card"><div class="stat-number">{total_units:,}</div><div class="stat-label">Total Units</div></div>""", unsafe_allow_html=True)

    with sc4:
        unique_boroughs = int(filtered_df["borough"].nunique()) if "borough" in filtered_df.columns else 0
        st.markdown(f"""<div class="stat-card"><div class="stat-number">{unique_boroughs}</div><div class="stat-label">Boroughs</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📋 List View", "🏢 Unit Distribution"])

    with tab1:
        st.markdown("### Housing Lottery Locations")
        st.caption("Click on markers for details. Green = Open, Red = Filled, Blue = Other")
        if not filtered_df.empty:
            lottery_map = create_map(filtered_df)
            st_folium(lottery_map, height=500)
        else:
            st.info("No lotteries found matching your criteria.")

    with tab2:
        st.markdown("### Housing Lottery Calendar")
        if filtered_df.empty:
            st.info("No lotteries found matching your criteria.")
        else:
            sort_col1, sort_col2 = st.columns([2, 2])
            with sort_col1:
                show_open_first = st.checkbox("Show open lotteries first", value=True)
            with sort_col2:
                view_mode = st.radio("View Mode", ["Card View", "Detailed View", "Table View"], horizontal=True)

            sorted_df = filtered_df.sort_values("lottery_end_date", ascending=True) if "lottery_end_date" in filtered_df.columns else filtered_df.copy()
            if show_open_first and "lottery_status" in sorted_df.columns:
                is_open = _safe_str_series(sorted_df, "lottery_status").str.contains("Open", case=False, na=False)
                sorted_df = pd.concat([sorted_df[is_open], sorted_df[~is_open]])

            items_per_page = 10
            total_pages = max(1, (len(sorted_df) - 1) // items_per_page + 1)
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="list_page_number")
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_df = sorted_df.iloc[start_idx:end_idx]

            if view_mode == "Card View":
                for _, row in page_df.iterrows():
                    display_lottery_card(row)
            elif view_mode == "Detailed View":
                for _, row in page_df.iterrows():
                    display_detailed_lottery_info(row)
            else:
                table_df = page_df.copy()
                rename_dict = {col: COLUMN_DEFINITIONS[col]["display_name"] for col in table_df.columns if col in COLUMN_DEFINITIONS}
                table_df = table_df.rename(columns=rename_dict)
                st.dataframe(table_df, width="stretch", height=420)

            st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(sorted_df))} of {len(sorted_df)} lotteries")
            st.markdown("---")
            st.download_button(
                "📥 Download All Filtered Data (CSV)",
                data=convert_df_to_csv(filtered_df),
                file_name=f"nyc_housing_lotteries_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_all",
            )

    with tab3:
        render_unit_distribution_tab(df)

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
      <p>Data source: <a href="https://data.cityofnewyork.us/Housing-Development/Advertised-Lotteries-on-Housing-Connect-By-Lottery/vy5i-a666" target="_blank">
      NYC Open Data - Housing Connect Lotteries</a></p>
      <p>For official applications, visit <a href="https://housingconnect.nyc.gov" target="_blank">NYC Housing Connect</a></p>
    </div>
""",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
