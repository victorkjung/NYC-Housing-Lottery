
"""
NYC Housing Lottery Finder
A Streamlit app to browse, map, and analyze NYC affordable housing lotteries.
- Adds schema-aware enrichment so List View + Unit Distribution use the same normalized bedroom-count fields.
- Adds Unit Distribution date-range presets + slider + trend analysis over time.
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
# Canonical unit size catalog (labels shown in UI)
# ---------------------------
UNIT_SIZE_LABELS = ["Studio", "1BR", "2BR", "3BR", "4+BR"]

# Candidate column names (NYC Open Data sometimes changes field names)
UNIT_SIZE_CANDIDATES: Dict[str, List[str]] = {
    "Studio": ["unit_distribution_studio", "unit_distribution_studios", "studio_units", "studios"],
    "1BR": ["unit_distribution_1_bedroom", "unit_distribution_1_bedrooms", "unit_distribution_1bed", "unit_distribution_1bedroom", "1_bedroom_units", "one_bedroom_units", "unit_1br"],
    "2BR": ["unit_distribution_2_bedrooms", "unit_distribution_2_bedroom", "unit_distribution_2bed", "unit_distribution_2bedroom", "2_bedroom_units", "two_bedroom_units", "unit_2br"],
    "3BR": ["unit_distribution_3_bedrooms", "unit_distribution_3_bedroom", "unit_distribution_3bed", "unit_distribution_3bedroom", "3_bedroom_units", "three_bedroom_units", "unit_3br"],
    "4+BR": ["unit_distribution_4_bedroom", "unit_distribution_4_bedrooms", "unit_distribution_4bed", "unit_distribution_4bedroom", "4_bedroom_units", "four_bedroom_units", "unit_4br", "unit_distribution_4_plus"],
}

# Normalized columns added by enrich_dataframe()
NORM_UNIT_COLS: Dict[str, str] = {
    "Studio": "unit_size_studio",
    "1BR": "unit_size_1br",
    "2BR": "unit_size_2br",
    "3BR": "unit_size_3br",
    "4+BR": "unit_size_4br_plus",
}

# ---------------------------
# Light CSS (keeps your dark theme-friendly spacing)
# ---------------------------
st.markdown(
    """
<style>
  @media (max-width: 768px) {
    .block-container { padding: 1rem 0.5rem; }
    .stSelectbox, .stDateInput { min-width: 100%; }
  }
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
def _to_timestamp(d: Optional[date | datetime]) -> Optional[pd.Timestamp]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return pd.Timestamp(d.date())
    return pd.Timestamp(d)

def _safe_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype="object")
    return df[col].astype(str)

def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

# ---------------------------
# Data fetch + enrichment
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_lottery_data() -> pd.DataFrame:
    api_url = "https://data.cityofnewyork.us/resource/vy5i-a666.json"
    params = {"$limit": 5000, "$order": "lottery_end_date DESC"}
    r = requests.get(api_url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()) if r.json() else pd.DataFrame()

    # Parse dates (only known fields; avoid accidentally parsing random strings)
    for col in ["lottery_start_date", "lottery_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure numeric types where possible
    for col in ["latitude", "longitude", "unit_count", "building_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize bedroom-count columns ONCE, so every tab uses the same truth."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Ensure canonical ID column is numeric-ish (helps joins/filters)
    if "lottery_id" in out.columns:
        out["lottery_id"] = pd.to_numeric(out["lottery_id"], errors="coerce")

    # Create normalized unit-size columns
    for label, norm_col in NORM_UNIT_COLS.items():
        candidates = UNIT_SIZE_CANDIDATES.get(label, [])
        actual = first_existing(out, candidates)
        if actual and actual in out.columns:
            out[norm_col] = safe_num(out[actual]).astype(int)
        else:
            out[norm_col] = 0

    return out

# ---------------------------
# Filtering (shared)
# ---------------------------
def filter_data(
    df: pd.DataFrame,
    borough: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[date | datetime] = None,
    end_date: Optional[date | datetime] = None,
    development_type: Optional[str] = None,
    date_field: str = "lottery_start_date",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    f = df.copy()

    if borough and borough != "All Boroughs" and "borough" in f.columns:
        f = f[f["borough"].astype(str).str.upper() == borough.upper()]

    if status and status != "All Statuses" and "lottery_status" in f.columns:
        f = f[f["lottery_status"].astype(str).str.contains(status, case=False, na=False)]

    if development_type and development_type != "All Types" and "development_type" in f.columns:
        f = f[f["development_type"].astype(str).str.contains(development_type, case=False, na=False)]

    sd = _to_timestamp(start_date)
    ed = _to_timestamp(end_date)

    if date_field in f.columns:
        if sd is not None:
            f = f[f[date_field] >= sd]
        if ed is not None:
            f = f[f[date_field] <= ed]

    return f

# ---------------------------
# Map
# ---------------------------
def create_map(df: pd.DataFrame) -> folium.Map:
    center_lat, center_lon = 40.7128, -74.0060
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    if df is None or df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return m

    map_df = df.dropna(subset=["latitude", "longitude"]).copy()
    if map_df.empty:
        return m

    # Center on filtered points
    center_lat = float(pd.to_numeric(map_df["latitude"], errors="coerce").dropna().mean())
    center_lon = float(pd.to_numeric(map_df["longitude"], errors="coerce").dropna().mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    for _, row in map_df.iterrows():
        status = str(row.get("lottery_status", "")).lower()
        color = "green" if "open" in status else ("red" if "filled" in status else "blue")

        lat = pd.to_numeric(row.get("latitude", None), errors="coerce")
        lon = pd.to_numeric(row.get("longitude", None), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue

        name = row.get("lottery_name", "Housing Lottery")
        popup = folium.Popup(f"<b>{name}</b><br>Status: {row.get('lottery_status','')}<br>Borough: {row.get('borough','')}<br>Total Units: {row.get('unit_count','')}", max_width=320)

        folium.Marker(
            location=[float(lat), float(lon)],
            popup=popup,
            icon=folium.Icon(color=color, icon="home", prefix="fa"),
            tooltip=name,
        ).add_to(m)

    return m

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# Unit Distribution Tab (with date presets + slider + trend)
# ---------------------------
def render_unit_distribution_tab(df_enriched: pd.DataFrame) -> None:
    st.markdown("## 🏢 Unit Distribution Analysis")
    st.caption("Analyze bedroom mix across lotteries (uses the same enriched dataframe as List View).")

    # Filters row (borough / status / type)
    c1, c2, c3 = st.columns(3)
    with c1:
        boroughs = ["All Boroughs"] + (sorted(df_enriched["borough"].dropna().unique().tolist()) if "borough" in df_enriched.columns else [])
        f_borough = st.selectbox("Borough", boroughs, key="ud_borough")
    with c2:
        f_status = st.selectbox("Status", ["All Statuses", "Open", "Closed", "Filled", "Active"], key="ud_status")
    with c3:
        types = ["All Types"] + (sorted(df_enriched["development_type"].dropna().unique().tolist()) if "development_type" in df_enriched.columns else [])
        f_type = st.selectbox("Development Type", types, key="ud_type")

    # Apply filters (same logic as List View uses)
    filtered = filter_data(df_enriched, borough=f_borough, status=f_status, development_type=f_type)
    if filtered.empty:
        st.warning('No data available for the selected filters.')
        return

    # ---- Date controls (preset chips + optional custom calendars) ----
    st.markdown("#### Date range")

    # Use lottery_start_date as the time axis
    date_col = "lottery_start_date"
    df_dates = filtered.dropna(subset=[date_col]).copy()

    if df_dates.empty:
        st.info("No valid lottery_start_date values available for date-based unit distribution.")
        df_time = filtered.copy()
        start_ts = end_ts = None
        preset = "All time"
    else:
        min_dt = pd.to_datetime(df_dates[date_col], errors="coerce").min()
        max_dt = pd.to_datetime(df_dates[date_col], errors="coerce").max()
        min_dt = pd.Timestamp(min_dt).normalize() if pd.notna(min_dt) else None
        max_dt = pd.Timestamp(max_dt).normalize() if pd.notna(max_dt) else None

        if min_dt is None or max_dt is None:
            st.info("No valid date range available for lottery_start_date.")
            df_time = filtered.copy()
            start_ts = end_ts = None
            preset = "All time"
        else:
            presets = ["Last 6 months", "Last 1 year", "Last 3 years", "All time", "Custom"]

            # "Chips" UI: prefer segmented_control if available, else fallback to horizontal radio
            if hasattr(st, "segmented_control"):
                preset = st.segmented_control("Preset", presets, default=presets[0], key="ud_date_preset_chip")
            else:
                preset = st.radio("Preset", presets, horizontal=True, index=0, key="ud_date_preset_chip")

            def _preset_start(p: str, maxd: pd.Timestamp, mind: pd.Timestamp) -> pd.Timestamp:
                if p == "Last 6 months":
                    return max(mind, (maxd - pd.DateOffset(months=6)).normalize())
                if p == "Last 1 year":
                    return max(mind, (maxd - pd.DateOffset(years=1)).normalize())
                if p == "Last 3 years":
                    return max(mind, (maxd - pd.DateOffset(years=3)).normalize())
                return mind  # All time

            if preset == "Custom":
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    custom_start = st.date_input(
                        "Start date (lottery_start_date)",
                        value=min_dt.date(),
                        min_value=min_dt.date(),
                        max_value=max_dt.date(),
                        key="ud_custom_start",
                    )
                with dcol2:
                    custom_end = st.date_input(
                        "End date (lottery_start_date)",
                        value=max_dt.date(),
                        min_value=min_dt.date(),
                        max_value=max_dt.date(),
                        key="ud_custom_end",
                    )
                start_ts = pd.Timestamp(custom_start).normalize()
                end_ts = pd.Timestamp(custom_end).normalize()
                if start_ts > end_ts:
                    st.warning("Start date is after end date. Swapping them.")
                    start_ts, end_ts = end_ts, start_ts
            else:
                start_ts = _preset_start(preset, max_dt, min_dt)
                end_ts = max_dt

            df_time = df_dates[(df_dates[date_col] >= start_ts) & (df_dates[date_col] <= end_ts)].copy()

    if start_ts is not None and end_ts is not None:

        st.caption(f"Unit Distribution range: **{start_ts.date()} → {end_ts.date()}** (lottery_start_date)")

    if df_time.empty:
        st.info("No lotteries match the selected filters/date range.")
        return

    # Unit sizes selector (always show all 5)
    selected_sizes = st.multiselect(
        "Unit sizes to include",
        options=UNIT_SIZE_LABELS,
        default=UNIT_SIZE_LABELS,
        key="ud_sizes",
    )

    chart_mode = st.radio("Chart view", ["Both", "Pie", "Bar"], horizontal=True, key="ud_chart_mode")
    include_zeros = st.checkbox("Include zero categories in charts", value=False, key="ud_include_zeros")

    # Summary totals using normalized columns
    totals = []
    for label in selected_sizes:
        norm_col = NORM_UNIT_COLS[label]
        totals.append(float(df_time[norm_col].sum()))

    summary_df = pd.DataFrame({"Unit Size": selected_sizes, "Units": totals})
    # Friendly order
    order = {"Studio": 0, "1BR": 1, "2BR": 2, "3BR": 3, "4+BR": 4}
    summary_df["__o"] = summary_df["Unit Size"].map(lambda x: order.get(x, 99))
    summary_df = summary_df.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    total_units = float(summary_df["Units"].sum())
    summary_df["Share"] = (summary_df["Units"] / total_units).map(lambda x: f"{x:.1%}") if total_units > 0 else "0.0%"

    st.markdown("### Summary")
    st.dataframe(summary_df, width="stretch", height=230)

    chart_df = summary_df.copy()
    if not include_zeros:
        chart_df = chart_df[chart_df["Units"] > 0].copy()

    if chart_df.empty:
        st.info("All selected unit sizes are 0 for this slice.")
    else:
        cc1, cc2 = st.columns(2)
        if chart_mode in ("Both", "Pie"):
            with cc1:
                fig_pie = px.pie(chart_df, values="Units", names="Unit Size", title="Unit Distribution by Size", hole=0.35)
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, width="stretch")
        if chart_mode in ("Both", "Bar"):
            with (cc2 if chart_mode == "Both" else cc1):
                fig_bar = px.bar(chart_df, x="Unit Size", y="Units", title="Total Units by Size", labels={"Units": "Number of Units"})
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, width="stretch")

    # ---- Trend shifts over time (monthly) ----
    st.markdown("### 📈 Trend shifts over time")
    if "lottery_start_date" not in df_time.columns or df_time["lottery_start_date"].dropna().empty:
        st.info("No lottery_start_date values available to plot trends.")
    else:
        trend = df_time.dropna(subset=["lottery_start_date"]).copy()
        trend["period"] = trend["lottery_start_date"].dt.to_period("M").astype(str)

        y_cols = [NORM_UNIT_COLS[s] for s in selected_sizes]
        grouped = trend.groupby("period")[y_cols].sum().reset_index()

        # Rename back to friendly labels for plotting legend
        rename = {NORM_UNIT_COLS[s]: s for s in selected_sizes}
        grouped = grouped.rename(columns=rename)

        if len(grouped) < 2:
            st.info("Not enough time periods in the current range to show a trend.")
        else:
            fig_line = px.line(grouped, x="period", y=selected_sizes, title="Monthly Unit Counts by Size", markers=True)
            fig_line.update_layout(xaxis_title="Month", yaxis_title="Units", xaxis_tickangle=-45)
            st.plotly_chart(fig_line, width="stretch")

            # Optional: show share (mix) trend
            share = grouped.copy()
            share_total = share[selected_sizes].sum(axis=1).replace({0: pd.NA})
            for s in selected_sizes:
                share[s] = (share[s] / share_total) * 100
            share = share.dropna()
            if len(share) > 1:
                fig_share = px.line(share, x="period", y=selected_sizes, title="Monthly Unit Mix (Share %) by Size", markers=True)
                fig_share.update_layout(xaxis_title="Month", yaxis_title="Share (%)", xaxis_tickangle=-45)
                st.plotly_chart(fig_share, width="stretch")

    # ---- Detailed table (uses normalized columns, so values match List View) ----
    st.markdown("### Detailed Data")
    base_cols = [c for c in ["lottery_id", "lottery_name", "borough", "lottery_status", "development_type", "unit_count"] if c in df_time.columns]
    detail = df_time[base_cols].copy()
    for label in UNIT_SIZE_LABELS:
        detail[f"{label} Units"] = df_time[NORM_UNIT_COLS[label]].astype(int)

    st.dataframe(detail, width="stretch", height=350)
    st.download_button(
        "📥 Download Unit Distribution (CSV)",
        data=convert_df_to_csv(detail),
        file_name=f"unit_distribution_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="ud_dl",
    )

# ---------------------------
# Main app
# ---------------------------
def main() -> None:
    st.title("🏠 NYC Housing Lottery Finder")

    with st.spinner("Loading lottery data..."):
        raw = fetch_lottery_data()
        df = enrich_dataframe(raw)

    if df.empty:
        st.error("Unable to load lottery data.")
        return

    # Global filters (used for Map + List)
    st.markdown("### 🔍 Filter Lotteries")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        boroughs = ["All Boroughs"] + (sorted(df["borough"].dropna().unique().tolist()) if "borough" in df.columns else [])
        selected_borough = st.selectbox("Borough", boroughs, key="main_borough")
    with col2:
        selected_status = st.selectbox("Status", ["All Statuses", "Open", "Closed", "Filled", "Active"], key="main_status")
    with col3:
        start_date = st.date_input("From Date", value=(datetime.now() - timedelta(days=30)).date(), key="main_start")
    with col4:
        end_date = st.date_input("To Date", value=(datetime.now() + timedelta(days=180)).date(), key="main_end")

    filtered_df = filter_data(df, borough=selected_borough, status=selected_status, start_date=start_date, end_date=end_date, date_field="lottery_start_date")

    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📋 List View", "🏢 Unit Distribution"])

    with tab1:
        st.subheader("Housing Lottery Locations")
        if filtered_df.empty:
            st.info("No lotteries match your filters.")
        else:
            st_folium(create_map(filtered_df), height=520)

    with tab2:
        st.subheader("Housing Lottery Calendar")
        if filtered_df.empty:
            st.info("No lotteries match your filters.")
        else:
            view_mode = st.radio("View Mode", ["Table View"], horizontal=True, key="lv_mode")
            table = filtered_df.copy()

            # Show normalized unit columns too (so user can verify alignment)
            for label in UNIT_SIZE_LABELS:
                table[f"{label} Units"] = table[NORM_UNIT_COLS[label]].astype(int)

            # Keep columns readable
            cols = [c for c in ["lottery_id", "lottery_name", "lottery_status", "development_type", "lottery_start_date", "lottery_end_date", "unit_count"] if c in table.columns]
            cols += [f"{label} Units" for label in UNIT_SIZE_LABELS]
            st.dataframe(table[cols], width="stretch", height=520)

            st.download_button(
                "📥 Download Filtered Data (CSV)",
                data=convert_df_to_csv(table[cols]),
                file_name=f"nyc_housing_lotteries_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="lv_dl",
            )

    with tab3:
        render_unit_distribution_tab(df)

    st.caption("Data source: NYC Open Data – Advertised Lotteries on Housing Connect (vy5i-a666)")

if __name__ == "__main__":
    main()
