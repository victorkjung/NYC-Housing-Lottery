"""
NYC Housing Lottery Finder
A Streamlit app to browse, map, and analyze NYC affordable housing lotteries.

Key guarantees:
- One single enriched dataframe (normalized unit-size fields) used by BOTH List View and Unit Distribution.
- Unit Distribution includes preset chips + optional custom date range + trend analysis (monthly totals + mix).
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import folium
from streamlit_folium import st_folium
import plotly.express as px
from typing import Optional, List, Dict


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
    "1BR": [
        "unit_distribution_1_bedroom",
        "unit_distribution_1_bedrooms",
        "unit_distribution_1bed",
        "unit_distribution_1bedroom",
        "1_bedroom_units",
        "one_bedroom_units",
        "unit_1br",
    ],
    "2BR": [
        "unit_distribution_2_bedrooms",
        "unit_distribution_2_bedroom",
        "unit_distribution_2bed",
        "unit_distribution_2bedroom",
        "2_bedroom_units",
        "two_bedroom_units",
        "unit_2br",
    ],
    "3BR": [
        "unit_distribution_3_bedrooms",
        "unit_distribution_3_bedroom",
        "unit_distribution_3bed",
        "unit_distribution_3bedroom",
        "3_bedroom_units",
        "three_bedroom_units",
        "unit_3br",
    ],
    "4+BR": [
        "unit_distribution_4_bedroom",
        "unit_distribution_4_bedrooms",
        "unit_distribution_4bed",
        "unit_distribution_4bedroom",
        "4_bedroom_units",
        "four_bedroom_units",
        "unit_4br",
        "unit_distribution_4_plus",
    ],
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
# CSS
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
    box-shadow: 0 4px 15px rgba(0,0,0,0.12);
  }
  .lottery-card h3 { margin: 0 0 0.5rem 0; font-size: 1.1rem; color: white; }
  .lottery-card p { margin: 0.25rem 0; font-size: 0.92rem; opacity: 0.95; }

  .status-open  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
  .status-closed{ background: linear-gradient(135deg, #636363 0%, #a2ab58 100%); }
  .status-filled{ background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }

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

def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# Data fetch + enrichment
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_lottery_data() -> pd.DataFrame:
    api_url = "https://data.cityofnewyork.us/resource/vy5i-a666.json"
    params = {"$limit": 5000, "$order": "lottery_end_date DESC"}
    r = requests.get(api_url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data) if data else pd.DataFrame()

    # Parse dates
    for col in ["lottery_start_date", "lottery_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure numeric types where possible
    for col in ["latitude", "longitude", "unit_count", "building_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Also coerce any candidate unit distribution columns that exist
    for label, candidates in UNIT_SIZE_CANDIDATES.items():
        for c in candidates:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize bedroom-count columns ONCE, so every tab uses the same truth."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Normalize lottery_id for stable joins/filters
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
# List view rendering helpers
# ---------------------------
def get_status_class(status: str) -> str:
    s = (status or "").lower()
    if "open" in s:
        return "status-open"
    if "filled" in s:
        return "status-filled"
    return "status-closed"

def display_lottery_card(row: pd.Series) -> None:
    status_class = get_status_class(str(row.get("lottery_status", "")))
    start_dt = pd.to_datetime(row.get("lottery_start_date", pd.NaT), errors="coerce")
    end_dt = pd.to_datetime(row.get("lottery_end_date", pd.NaT), errors="coerce")
    start_str = start_dt.strftime("%b %d, %Y") if pd.notna(start_dt) else "TBD"
    end_str = end_dt.strftime("%b %d, %Y") if pd.notna(end_dt) else "TBD"

    st.markdown(
        f"""
<div class="lottery-card {status_class}">
  <h3>🏠 {row.get('lottery_name', 'N/A')}</h3>
  <p><strong>🆔 Lottery ID:</strong> {row.get('lottery_id', 'N/A')}</p>
  <p><strong>📍 Borough:</strong> {row.get('borough', 'N/A')}</p>
  <p><strong>📋 Status:</strong> {row.get('lottery_status', 'N/A')}</p>
  <p><strong>🏗️ Type:</strong> {row.get('development_type', 'N/A')}</p>
  <p><strong>🏢 Total Units:</strong> {int(pd.to_numeric(row.get('unit_count', 0), errors='coerce') or 0):,}</p>
  <p><strong>🛏️ Unit mix:</strong> Studio {int(row.get(NORM_UNIT_COLS['Studio'], 0))}, 1BR {int(row.get(NORM_UNIT_COLS['1BR'], 0))}, 2BR {int(row.get(NORM_UNIT_COLS['2BR'], 0))}, 3BR {int(row.get(NORM_UNIT_COLS['3BR'], 0))}, 4+BR {int(row.get(NORM_UNIT_COLS['4+BR'], 0))}</p>
  <p><strong>📅 Application Period:</strong> {start_str} – {end_str}</p>
</div>
""",
        unsafe_allow_html=True,
    )

def display_detailed_lottery_info(row: pd.Series) -> None:
    title = f"🏠 {row.get('lottery_name', 'N/A')} — {row.get('lottery_status', 'N/A')}"
    with st.expander(title, expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Basic")
            st.write("**Lottery ID:**", row.get("lottery_id", "N/A"))
            st.write("**Borough:**", row.get("borough", "N/A"))
            st.write("**Status:**", row.get("lottery_status", "N/A"))
            st.write("**Type:**", row.get("development_type", "N/A"))
            st.write("**Total Units:**", int(pd.to_numeric(row.get("unit_count", 0), errors="coerce") or 0))

            sd = pd.to_datetime(row.get("lottery_start_date", pd.NaT), errors="coerce")
            ed = pd.to_datetime(row.get("lottery_end_date", pd.NaT), errors="coerce")
            st.write("**Start Date:**", sd.strftime("%m/%d/%Y") if pd.notna(sd) else "N/A")
            st.write("**End Date:**", ed.strftime("%m/%d/%Y") if pd.notna(ed) else "N/A")

        with c2:
            st.markdown("#### Unit distribution (normalized)")
            st.write("**Studio:**", int(row.get(NORM_UNIT_COLS["Studio"], 0)))
            st.write("**1BR:**", int(row.get(NORM_UNIT_COLS["1BR"], 0)))
            st.write("**2BR:**", int(row.get(NORM_UNIT_COLS["2BR"], 0)))
            st.write("**3BR:**", int(row.get(NORM_UNIT_COLS["3BR"], 0)))
            st.write("**4+BR:**", int(row.get(NORM_UNIT_COLS["4+BR"], 0)))

        with c3:
            st.markdown("#### Location")
            st.write("**Zip:**", row.get("postcode", "N/A"))
            st.write("**Community Board:**", row.get("community_board", "N/A"))
            st.write("**Latitude:**", row.get("latitude", "N/A"))
            st.write("**Longitude:**", row.get("longitude", "N/A"))

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
        popup = folium.Popup(
            f"<b>{name}</b><br>"
            f"Status: {row.get('lottery_status','')}<br>"
            f"Borough: {row.get('borough','')}<br>"
            f"Total Units: {row.get('unit_count','')}",
            max_width=320,
        )

        folium.Marker(
            location=[float(lat), float(lon)],
            popup=popup,
            icon=folium.Icon(color=color, icon="home", prefix="fa"),
            tooltip=name,
        ).add_to(m)

    return m

# ---------------------------
# Unit Distribution Tab (preset chips + custom calendars + trend)
# ---------------------------
def render_unit_distribution_tab(df_enriched: pd.DataFrame) -> None:
    st.markdown("## 🏢 Unit Distribution Analysis")
    st.caption("Bedroom mix across lotteries (same enriched dataframe as List View).")

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

    filtered = filter_data(df_enriched, borough=f_borough, status=f_status, development_type=f_type, date_field="lottery_start_date")
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    # ---- Date controls (chips + optional custom calendars) ----
    st.markdown("#### Date range (lottery_start_date)")

    date_col = "lottery_start_date"
    df_dates = filtered.dropna(subset=[date_col]).copy()

    if df_dates.empty:
        st.info("No valid lottery_start_date values available for date-based unit distribution.")
        df_time = filtered.copy()
        start_ts = end_ts = None
    else:
        min_dt = pd.to_datetime(df_dates[date_col], errors="coerce").min()
        max_dt = pd.to_datetime(df_dates[date_col], errors="coerce").max()
        min_dt = pd.Timestamp(min_dt).normalize() if pd.notna(min_dt) else None
        max_dt = pd.Timestamp(max_dt).normalize() if pd.notna(max_dt) else None

        if min_dt is None or max_dt is None:
            st.info("No valid date range available.")
            df_time = filtered.copy()
            start_ts = end_ts = None
        else:
            presets = ["Last 6 months", "Last 1 year", "Last 3 years", "All time", "Custom"]
            if hasattr(st, "segmented_control"):
                preset = st.segmented_control("Preset", presets, default=presets[0], key="ud_preset")
            else:
                preset = st.radio("Preset", presets, horizontal=True, index=0, key="ud_preset")

            def preset_start(p: str, maxd: pd.Timestamp, mind: pd.Timestamp) -> pd.Timestamp:
                if p == "Last 6 months":
                    return max(mind, (maxd - pd.DateOffset(months=6)).normalize())
                if p == "Last 1 year":
                    return max(mind, (maxd - pd.DateOffset(years=1)).normalize())
                if p == "Last 3 years":
                    return max(mind, (maxd - pd.DateOffset(years=3)).normalize())
                return mind

            if preset == "Custom":
                d1, d2 = st.columns(2)
                with d1:
                    custom_start = st.date_input("Start date", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date(), key="ud_custom_start")
                with d2:
                    custom_end = st.date_input("End date", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date(), key="ud_custom_end")
                start_ts = pd.Timestamp(custom_start).normalize()
                end_ts = pd.Timestamp(custom_end).normalize()
                if start_ts > end_ts:
                    start_ts, end_ts = end_ts, start_ts
            else:
                start_ts = preset_start(preset, max_dt, min_dt)
                end_ts = max_dt

            df_time = df_dates[(df_dates[date_col] >= start_ts) & (df_dates[date_col] <= end_ts)].copy()
            st.caption(f"Unit Distribution range: **{start_ts.date()} → {end_ts.date()}**")

    if df_time.empty:
        st.info("No lotteries match the selected filters/date range.")
        return

    # Unit sizes selector (always show all 5)
    selected_sizes = st.multiselect("Unit sizes to include", options=UNIT_SIZE_LABELS, default=UNIT_SIZE_LABELS, key="ud_sizes")
    chart_mode = st.radio("Chart view", ["Both", "Pie", "Bar"], horizontal=True, key="ud_chart_mode")
    include_zeros = st.checkbox("Include zero categories in charts", value=False, key="ud_include_zeros")

    # Summary totals using normalized columns
    summary_rows = []
    for label in selected_sizes:
        norm_col = NORM_UNIT_COLS[label]
        summary_rows.append({"Unit Size": label, "Units": float(df_time[norm_col].sum())})

    summary_df = pd.DataFrame(summary_rows)
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
        grouped = grouped.rename(columns={NORM_UNIT_COLS[s]: s for s in selected_sizes})

        if len(grouped) < 2:
            st.info("Not enough time periods in the current range to show a trend.")
        else:
            fig_line = px.line(grouped, x="period", y=selected_sizes, title="Monthly Unit Counts by Size", markers=True)
            fig_line.update_layout(xaxis_title="Month", yaxis_title="Units", xaxis_tickangle=-45)
            st.plotly_chart(fig_line, width="stretch")

            share = grouped.copy()
            share_total = share[selected_sizes].sum(axis=1).replace({0: pd.NA})
            for s in selected_sizes:
                share[s] = (share[s] / share_total) * 100
            share = share.dropna()
            if len(share) > 1:
                fig_share = px.line(share, x="period", y=selected_sizes, title="Monthly Unit Mix (Share %) by Size", markers=True)
                fig_share.update_layout(xaxis_title="Month", yaxis_title="Share (%)", xaxis_tickangle=-45)
                st.plotly_chart(fig_share, width="stretch")

    # ---- Detailed table ----
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

    # Global filters (Map + List)
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

    filtered_df = filter_data(
        df,
        borough=selected_borough,
        status=selected_status,
        start_date=start_date,
        end_date=end_date,
        development_type=None,
        date_field="lottery_start_date",
    )

    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📋 List View", "🏢 Unit Distribution"])

    with tab1:
        st.subheader("Housing Lottery Locations")
        if filtered_df.empty:
            st.info("No lotteries match your filters.")
        else:
            st_folium(create_map(filtered_df), height=520)

    with tab2:
        st.markdown("### Housing Lottery Calendar")
        st.markdown("Browse lotteries in **Card**, **Detailed**, or **Table** view. All views use the same enriched dataframe.")

        if filtered_df.empty:
            st.info("No lotteries found matching your criteria.")
        else:
            sort_col1, sort_col2 = st.columns([2, 2])
            with sort_col1:
                show_open_first = st.checkbox("Show open lotteries first", value=True, key="lv_open_first")
            with sort_col2:
                view_mode = st.radio("View Mode", ["Card View", "Detailed View", "Table View"], horizontal=True, key="lv_view_mode")

            sorted_df = filtered_df.sort_values("lottery_end_date", ascending=True) if "lottery_end_date" in filtered_df.columns else filtered_df.copy()

            if show_open_first and "lottery_status" in sorted_df.columns:
                is_open = sorted_df["lottery_status"].astype(str).str.contains("Open", case=False, na=False)
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
                st.dataframe(page_df, width="stretch", height=420)

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

    st.caption("Data source: NYC Open Data – Advertised Lotteries on Housing Connect (vy5i-a666)")

if __name__ == "__main__":
    main()
