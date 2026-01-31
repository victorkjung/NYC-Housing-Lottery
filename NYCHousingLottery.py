"""
NYC Housing Lottery Finder
A Streamlit app to browse, map, and analyze NYC affordable housing lotteries
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
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

# Canonical (used for conversion attempts; UI will be schema-aware)
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

# ✅ Always show these five unit sizes in the UI (even if missing in API response)
UNIT_SIZE_CATALOG: List[Dict[str, object]] = [
    {"key": "studio", "label": "Studio", "candidates": ["unit_distribution_studio", "studio_units", "studios"]},
    {"key": "1br", "label": "1BR", "candidates": ["unit_distribution_1_bedroom", "unit_distribution_1_bedrooms", "1_bedroom_units", "one_bedroom_units", "unit_1br"]},
    {"key": "2br", "label": "2BR", "candidates": ["unit_distribution_2_bedrooms", "unit_distribution_2_bedroom", "2_bedroom_units", "two_bedroom_units", "unit_2br"]},
    {"key": "3br", "label": "3BR", "candidates": ["unit_distribution_3_bedrooms", "unit_distribution_3_bedroom", "3_bedroom_units", "three_bedroom_units", "unit_3br"]},
    {"key": "4br", "label": "4+BR", "candidates": ["unit_distribution_4_bedroom", "unit_distribution_4_bedrooms", "4_bedroom_units", "four_bedroom_units", "unit_4br", "unit_distribution_4_plus"]},
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
# Schema resolution helpers
# ---------------------------
def resolve_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


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


def resolve_unit_distribution_columns(df: pd.DataFrame) -> Dict[str, Tuple[Optional[str], str]]:
    """
    Returns: key -> (actual_col_or_None, label)
    Always includes all 5 sizes (actual col may be None if missing).
    """
    resolved: Dict[str, Tuple[Optional[str], str]] = {}
    for item in UNIT_SIZE_CATALOG:
        key = str(item["key"])
        label = str(item["label"])
        candidates = list(item["candidates"])  # type: ignore
        actual = resolve_first_existing(df, candidates)
        resolved[key] = (actual, label)
    return resolved


def resolve_ami_columns(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    alias_map = {
        "eli": (["applied_income_ami_category_extremely_low_income", "extremely_low_income", "ami_extremely_low", "ami_0_30"], "Extremely Low (0–30% AMI)"),
        "vli": (["applied_income_ami_category_very_low_income", "very_low_income", "ami_very_low", "ami_31_50"], "Very Low (31–50% AMI)"),
        "li": (["applied_income_ami_category_low_income", "low_income", "ami_low", "ami_51_80"], "Low (51–80% AMI)"),
        "mod": (["applied_income_ami_category_moderate_income", "moderate_income", "ami_moderate", "ami_81_120"], "Moderate (81–120% AMI)"),
        "mid": (["applied_income_ami_category_middle_income", "middle_income", "ami_middle", "ami_121_165"], "Middle (121–165% AMI)"),
        "above": (["applied_income_ami_category_above_middle_income", "above_middle_income", "ami_above_middle", "ami_165_plus"], "Above Middle (165%+ AMI)"),
    }
    resolved: Dict[str, Tuple[str, str]] = {}
    for logical, (candidates, label) in alias_map.items():
        actual = resolve_first_existing(df, candidates)
        if actual:
            resolved[logical] = (actual, label)
    return resolved


def resolve_preference_pct_columns(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    alias_map = {
        "mobility": (["lottery_mobility_percentage", "mobility_percentage", "mobility_pct"], "Mobility"),
        "vh": (["lottery_vision_hearing_percentage", "vision_hearing_percentage", "vision_hearing_pct"], "Vision/Hearing"),
        "cb": (["lottery_community_board_percentage", "community_board_percentage", "community_board_pct"], "Community Board"),
        "mv": (["lottery_municipal_employee_military_veteran_percentage", "municipal_employee_military_veteran_percentage", "municipal_veteran_pct"], "Municipal/Veteran"),
        "nycha": (["lottery_nycha_percentage", "nycha_percentage", "nycha_pct"], "NYCHA"),
        "senior": (["lottery_senior_percentage", "senior_percentage", "senior_pct"], "Senior"),
    }
    resolved: Dict[str, Tuple[str, str]] = {}
    for logical, (candidates, label) in alias_map.items():
        actual = resolve_first_existing(df, candidates)
        if actual:
            resolved[logical] = (actual, label)
    return resolved


# ---------------------------
# General helpers
# ---------------------------
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

        # Convert date columns
        for col in ["lottery_start_date", "lottery_end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert numeric columns if present
        numeric_candidates = [
            "latitude",
            "longitude",
            "unit_count",
            "building_count",
            *UNIT_DIST_COLS,
            *AMI_COLS,
            *LOTTERY_PCT_COLS,
        ]
        for col in numeric_candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


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
        if "open" in status:
            color = "green"
        elif "filled" in status:
            color = "red"
        else:
            color = "blue"

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


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


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
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Basic Information")
            for key in [
                "lottery_id",
                "lottery_name",
                "lottery_status",
                "development_type",
                "lottery_start_date",
                "lottery_end_date",
                "building_count",
                "unit_count",
            ]:
                if key in row.index:
                    value = row.get(key, "N/A")
                    if pd.isna(value):
                        value = "N/A"
                    elif "date" in key and pd.notna(value):
                        value = pd.to_datetime(value).strftime("%m/%d/%Y")
                    label = COLUMN_DEFINITIONS.get(key, {}).get("display_name", key)
                    desc = COLUMN_DEFINITIONS.get(key, {}).get("description", "")
                    st.markdown(f"**{label}:** {value}")
                    if desc:
                        st.caption(desc)

        with col2:
            st.markdown("#### Unit Distribution")
            resolved_units = resolve_unit_distribution_columns(pd.DataFrame([row]))
            for _, (colname, label) in resolved_units.items():
                if colname and colname in row.index:
                    val = row.get(colname, 0)
                else:
                    val = 0
                val = 0 if pd.isna(val) else val
                st.markdown(f"**{label}:** {int(float(val))}")

            st.markdown("#### Location")
            for key in ["borough", "postcode", "community_board"]:
                if key in row.index:
                    label = COLUMN_DEFINITIONS.get(key, {}).get("display_name", key)
                    value = row.get(key, "N/A")
                    value = "N/A" if pd.isna(value) else value
                    st.markdown(f"**{label}:** {value}")

        with col3:
            st.markdown("#### AMI Categories")
            resolved_ami = resolve_ami_columns(pd.DataFrame([row]))
            if not resolved_ami:
                st.info("AMI category fields are not present for this record.")
            else:
                for _, (colname, label) in resolved_ami.items():
                    val = row.get(colname, 0)
                    val = 0 if pd.isna(val) else val
                    st.markdown(f"**{label}:** {int(float(val))}")

            st.markdown("#### Lottery Preferences (%)")
            resolved_pct = resolve_preference_pct_columns(pd.DataFrame([row]))
            if not resolved_pct:
                st.info("Preference % fields are not present for this record.")
            else:
                for _, (colname, label) in resolved_pct.items():
                    val = row.get(colname, 0)
                    val = 0 if pd.isna(val) else val
                    st.markdown(f"**{label}:** {float(val):.1f}%")


# ---------------------------
# Tabs
# ---------------------------
def render_unit_distribution_tab(df: pd.DataFrame) -> None:
    st.markdown("### 🏢 Unit Distribution Analysis")
    st.markdown("Analyze the distribution of unit sizes across housing lotteries.")

    st.markdown("#### Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        boroughs = ["All Boroughs"] + (sorted(df["borough"].dropna().unique().tolist()) if "borough" in df.columns else [])
        ud_borough = st.selectbox("Borough", boroughs, key="ud_borough")

    with filter_col2:
        ud_status = st.selectbox("Status", ["All Statuses", "Open", "Closed", "Filled"], key="ud_status")

    with filter_col3:
        types = ["All Types"] + (sorted(df["development_type"].dropna().unique().tolist()) if "development_type" in df.columns else [])
        ud_type = st.selectbox("Development Type", types, key="ud_type")

    filtered = filter_data(df, borough=ud_borough, status=ud_status, development_type=ud_type)
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    # ✅ Always resolve all 5 sizes; missing columns become zeros
    resolved = resolve_unit_distribution_columns(filtered)

    # Always show all five sizes in the filter
    all_labels = [label for (_, label) in resolved.values()]
    selected_labels = st.multiselect(
        "Unit sizes to include",
        options=all_labels,
        default=all_labels,
        help="All five unit sizes are shown. If a size is missing from the dataset response, it will display as 0.",
        key="ud_unit_sizes",
    )

    chart_mode = st.radio("Chart view", ["Both", "Pie", "Bar"], horizontal=True, key="ud_chart_mode")
    include_zeros_in_charts = st.checkbox(
        "Include zero categories in charts",
        value=False,
        help="Bar charts can include zeros; pie charts usually look better without zeros.",
        key="ud_include_zeros_in_charts",
    )

    # Build label->actual map (actual may be None)
    label_to_actual: Dict[str, Optional[str]] = {}
    for _, (actual, label) in resolved.items():
        label_to_actual[label] = actual

    # Totals (missing columns => 0)
    totals: Dict[str, float] = {}
    for lbl in selected_labels:
        actual = label_to_actual.get(lbl)
        if actual and actual in filtered.columns:
            totals[lbl] = safe_numeric_sum(filtered[actual])
        else:
            totals[lbl] = 0.0

    # Summary table should always show selected categories (including zeros)
    summary_df = (
        pd.DataFrame({"Unit Size": list(totals.keys()), "Units": list(totals.values())})
        .sort_values("Unit Size")
        .reset_index(drop=True)
    )

    # Friendly ordering Studio, 1BR, 2BR, 3BR, 4+BR
    order = {"Studio": 0, "1BR": 1, "2BR": 2, "3BR": 3, "4+BR": 4}
    summary_df["__order"] = summary_df["Unit Size"].map(lambda x: order.get(x, 99))
    summary_df = summary_df.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    total_units_all = float(summary_df["Units"].sum())
    if total_units_all > 0:
        summary_df["Share"] = (summary_df["Units"] / total_units_all).map(lambda x: f"{x:.1%}")
    else:
        summary_df["Share"] = "0.0%"

    st.markdown("#### Summary")
    st.dataframe(summary_df, width="stretch", height=230)

    # Prepare chart dataframe (optionally drop zeros)
    chart_df = summary_df.copy()
    if not include_zeros_in_charts:
        chart_df = chart_df[chart_df["Units"] > 0].copy()

    if chart_df.empty:
        st.info("All selected unit sizes are 0 for these filters (or missing in the dataset response).")
    else:
        chart_col1, chart_col2 = st.columns(2)

        if chart_mode in ("Both", "Pie"):
            with chart_col1:
                fig_pie = px.pie(
                    chart_df,
                    values="Units",
                    names="Unit Size",
                    title="Unit Distribution by Size",
                    hole=0.35,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, width="stretch")

        if chart_mode in ("Both", "Bar"):
            with (chart_col2 if chart_mode == "Both" else chart_col1):
                fig_bar = px.bar(
                    chart_df,
                    x="Unit Size",
                    y="Units",
                    title="Total Units by Size",
                    labels={"Units": "Number of Units"},
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, width="stretch")

    # Detailed table: always show all five columns (computed if missing)
    st.markdown("#### Detailed Data")
    base_cols = [c for c in ["lottery_name", "borough", "lottery_status", "development_type"] if c in filtered.columns]
    display_df = filtered[base_cols].copy()

    # Add computed columns for each selected label
    for lbl in selected_labels:
        actual = label_to_actual.get(lbl)
        out_col = f"{lbl} Units"
        if actual and actual in filtered.columns:
            display_df[out_col] = pd.to_numeric(filtered[actual], errors="coerce").fillna(0).astype(int)
        else:
            display_df[out_col] = 0

    # Rename base columns
    rename_map = {
        "lottery_name": "Lottery Name",
        "borough": "Borough",
        "lottery_status": "Status",
        "development_type": "Development Type",
    }
    display_df = display_df.rename(columns=rename_map)

    st.dataframe(display_df, width="stretch", height=350)
    st.download_button(
        "📥 Download Unit Distribution Data (CSV)",
        data=convert_df_to_csv(display_df),
        file_name=f"unit_distribution_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_unit_dist",
    )

    # Helpful diagnostics (collapsed)
    with st.expander("🔎 Diagnostics (optional)"):
        st.caption("If unit sizes are all 0, the dataset response may not contain those fields or they may be empty.")
        st.write("Detected unit-related columns:", detect_related_columns(filtered, ["unit", "bed", "studio"]))


def render_ami_category_tab(df: pd.DataFrame) -> None:
    st.markdown("### 💰 Applied Income AMI Category Analysis")
    st.markdown("Analyze units by Area Median Income (AMI) eligibility categories.")

    with st.expander("ℹ️ What are AMI Categories?"):
        st.markdown(
            """
**Area Median Income (AMI)** is used to determine eligibility for affordable housing:
- **Extremely Low Income**: 0–30% of AMI
- **Very Low Income**: 31–50% of AMI
- **Low Income**: 51–80% of AMI
- **Moderate Income**: 81–120% of AMI
- **Middle Income**: 121–165% of AMI
- **Above Middle Income**: 165%+ AMI
"""
        )

    st.markdown("#### Filters")
    c1, c2, c3 = st.columns(3)

    with c1:
        boroughs = ["All Boroughs"] + (sorted(df["borough"].dropna().unique().tolist()) if "borough" in df.columns else [])
        b = st.selectbox("Borough", boroughs, key="ami_borough")

    with c2:
        s = st.selectbox("Status", ["All Statuses", "Open", "Closed", "Filled"], key="ami_status")

    with c3:
        types = ["All Types"] + (sorted(df["development_type"].dropna().unique().tolist()) if "development_type" in df.columns else [])
        t = st.selectbox("Development Type", types, key="ami_type")

    filtered = filter_data(df, borough=b, status=s, development_type=t)
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    resolved = resolve_ami_columns(filtered)
    if not resolved:
        st.info("AMI category columns are not present in the dataset response.")
        candidates = detect_related_columns(filtered, ["ami", "income"])
        if candidates:
            st.caption("Possible AMI/income columns detected:")
            st.code(", ".join(candidates)[:2000])
        return

    labels = [label for (_, label) in resolved.values()]
    selected = st.multiselect("AMI categories to include", labels, default=labels, key="ami_select")
    label_to_actual = {label: actual for (_, (actual, label)) in resolved.items()}

    totals: Dict[str, float] = {}
    for lbl in selected:
        col = label_to_actual.get(lbl)
        if col and col in filtered.columns:
            totals[lbl] = safe_numeric_sum(filtered[col])

    totals_nonzero = {k: v for k, v in totals.items() if v > 0}
    if not totals_nonzero:
        st.info("No AMI category totals available (all zero/missing) for these filters.")
        return

    summary_df = (
        pd.DataFrame({"AMI Category": list(totals_nonzero.keys()), "Units": list(totals_nonzero.values())})
        .sort_values("Units", ascending=False)
        .reset_index(drop=True)
    )
    st.markdown("#### Summary")
    st.dataframe(summary_df, width="stretch", height=230)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = px.pie(summary_df, values="Units", names="AMI Category", title="Units by AMI Category", hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width="stretch")

    with chart_col2:
        figb = px.bar(summary_df, x="AMI Category", y="Units", title="Total Units by AMI Category")
        figb.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(figb, width="stretch")

    st.markdown("#### AMI Distribution Over Time")
    if "lottery_start_date" in filtered.columns:
        time_data = filtered.dropna(subset=["lottery_start_date"]).copy()
        if not time_data.empty:
            time_data["year_month"] = time_data["lottery_start_date"].dt.to_period("M").astype(str)
            cols = [label_to_actual[lbl] for lbl in selected if lbl in label_to_actual and label_to_actual[lbl] in filtered.columns]
            if cols:
                tg = time_data.groupby("year_month")[cols].sum(numeric_only=True).reset_index()
                rename = {"year_month": "Period"}
                for lbl in selected:
                    col = label_to_actual.get(lbl)
                    if col in tg.columns:
                        rename[col] = lbl
                tg = tg.rename(columns=rename)

                if len(tg) > 1:
                    figl = px.line(tg, x="Period", y=[c for c in tg.columns if c != "Period"], title="AMI Trends Over Time", markers=True)
                    figl.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(figl, width="stretch")
                else:
                    st.info("Not enough time periods to display a trend.")
        else:
            st.info("No valid start dates available to plot trends.")
    else:
        st.info("No start-date field available to plot trends.")

    st.markdown("#### Detailed Data")
    base_cols = [c for c in ["lottery_name", "borough", "lottery_status", "development_type"] if c in filtered.columns]
    cols = [label_to_actual[lbl] for lbl in selected if lbl in label_to_actual and label_to_actual[lbl] in filtered.columns]
    display_df = filtered[base_cols + cols].copy()

    rename_map = {"lottery_name": "Lottery Name", "borough": "Borough", "lottery_status": "Status", "development_type": "Development Type"}
    for lbl in selected:
        col = label_to_actual.get(lbl)
        if col in display_df.columns:
            rename_map[col] = lbl
    display_df = display_df.rename(columns=rename_map)

    st.dataframe(display_df, width="stretch", height=320)
    st.download_button(
        "📥 Download AMI Category Data (CSV)",
        data=convert_df_to_csv(display_df),
        file_name=f"ami_categories_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_ami",
    )


def render_lottery_percentage_tab(df: pd.DataFrame) -> None:
    st.markdown("### 📊 Lottery Preference Percentage Analysis")
    st.markdown("Analyze lottery preference allocations for special populations.")

    with st.expander("ℹ️ What are Lottery Preferences?"):
        st.markdown(
            """
NYC Housing Lotteries may reserve percentages of units for:
- **Mobility**: Applicants with mobility disabilities
- **Vision/Hearing**: Applicants with vision or hearing disabilities
- **Community Board**: Residents of the local community board district
- **Municipal/Veteran**: NYC municipal employees and military veterans
- **NYCHA**: Current NYCHA residents
- **Senior**: Seniors (typically 62+)
"""
        )

    st.markdown("#### Filters")
    c1, c2, c3 = st.columns(3)

    with c1:
        boroughs = ["All Boroughs"] + (sorted(df["borough"].dropna().unique().tolist()) if "borough" in df.columns else [])
        b = st.selectbox("Borough", boroughs, key="lp_borough")

    with c2:
        s = st.selectbox("Status", ["All Statuses", "Open", "Closed", "Filled"], key="lp_status")

    with c3:
        types = ["All Types"] + (sorted(df["development_type"].dropna().unique().tolist()) if "development_type" in df.columns else [])
        t = st.selectbox("Development Type", types, key="lp_type")

    filtered = filter_data(df, borough=b, status=s, development_type=t)
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    resolved = resolve_preference_pct_columns(filtered)
    if not resolved:
        st.info("Preference percentage columns are not present in the dataset response.")
        candidates = detect_related_columns(filtered, ["percent", "percentage", "pct", "preference", "senior", "nycha", "mobility"])
        if candidates:
            st.caption("Possible preference/percentage columns detected:")
            st.code(", ".join(candidates)[:2000])
        return

    labels = [label for (_, label) in resolved.values()]
    selected = st.multiselect("Preferences to include", labels, default=labels, key="lp_select")
    label_to_actual = {label: actual for (_, (actual, label)) in resolved.items()}

    avgs: Dict[str, float] = {}
    for lbl in selected:
        col = label_to_actual.get(lbl)
        if col and col in filtered.columns:
            av = safe_numeric_mean(filtered[col])
            if pd.notna(av):
                avgs[lbl] = av

    if not avgs:
        st.info("No preference percentage values available for these filters.")
        return

    summary_df = (
        pd.DataFrame({"Preference": list(avgs.keys()), "Avg %": list(avgs.values())})
        .sort_values("Avg %", ascending=False)
        .reset_index(drop=True)
    )

    st.markdown("#### Summary")
    st.dataframe(summary_df, width="stretch", height=230)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = px.pie(summary_df, values="Avg %", names="Preference", title="Average Preference % (Share)", hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width="stretch")

    with chart_col2:
        figb = px.bar(summary_df, x="Preference", y="Avg %", title="Average Preference %", labels={"Avg %": "Average %"})
        figb.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(figb, width="stretch")

    st.markdown("#### Detailed Data")
    base_cols = [c for c in ["lottery_name", "borough", "lottery_status", "development_type"] if c in filtered.columns]
    cols = [label_to_actual[lbl] for lbl in selected if lbl in label_to_actual and label_to_actual[lbl] in filtered.columns]
    display_df = filtered[base_cols + cols].copy()

    rename_map = {"lottery_name": "Lottery Name", "borough": "Borough", "lottery_status": "Status", "development_type": "Development Type"}
    for lbl in selected:
        col = label_to_actual.get(lbl)
        if col in display_df.columns:
            rename_map[col] = f"{lbl} %"
    display_df = display_df.rename(columns=rename_map)

    st.dataframe(display_df, width="stretch", height=320)
    st.download_button(
        "📥 Download Lottery Preferences Data (CSV)",
        data=convert_df_to_csv(display_df),
        file_name=f"lottery_preferences_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_lottery_pct",
    )


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

    with st.spinner("Loading lottery data..."):
        df = fetch_lottery_data()

    if df.empty:
        st.error("Unable to load lottery data. Please try again later.")
        return

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
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.markdown(
            f"""
        <div class="stat-card">
            <div class="stat-number">{len(filtered_df)}</div>
            <div class="stat-label">Total Lotteries</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with stat_col2:
        if "lottery_status" in filtered_df.columns:
            open_count = int(_safe_str_series(filtered_df, "lottery_status").str.contains("Open", case=False, na=False).sum())
        else:
            open_count = 0
        st.markdown(
            f"""
        <div class="stat-card">
            <div class="stat-number">{open_count}</div>
            <div class="stat-label">Currently Open</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with stat_col3:
        total_units = int(pd.to_numeric(filtered_df["unit_count"], errors="coerce").fillna(0).sum()) if "unit_count" in filtered_df.columns else 0
        st.markdown(
            f"""
        <div class="stat-card">
            <div class="stat-number">{total_units:,}</div>
            <div class="stat-label">Total Units</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with stat_col4:
        unique_boroughs = int(filtered_df["borough"].nunique()) if "borough" in filtered_df.columns else 0
        st.markdown(
            f"""
        <div class="stat-card">
            <div class="stat-number">{unique_boroughs}</div>
            <div class="stat-label">Boroughs</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🗺️ Map View", "📋 List View", "🏢 Unit Distribution", "💰 AMI Categories", "📊 Lottery Preferences"]
    )

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
        st.markdown("Click on each lottery to view details (units, AMI, preferences if available).")

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

    with tab4:
        render_ami_category_tab(df)

    with tab5:
        render_lottery_percentage_tab(df)

    with st.expander("📖 Column Reference Guide"):
        ref_df = pd.DataFrame(
            [{"Field Name": c, "Display Name": info["display_name"], "Description": info["description"]} for c, info in COLUMN_DEFINITIONS.items()]
        )
        st.dataframe(ref_df, width="stretch", height=420)
        st.caption("If a tab says fields are missing, compare these names against df.columns from the dataset response.")

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
```
