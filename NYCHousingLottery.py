"""
NYC Housing Lottery Finder
A Streamlit app to browse and map NYC affordable housing lotteries
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="NYC Housing Lottery Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile-friendly design
st.markdown("""
<style>
    /* Mobile-responsive adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem;
        }
        .stSelectbox, .stDateInput {
            min-width: 100%;
        }
    }
    
    /* Card styling for lottery items */
    .lottery-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .lottery-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        color: white;
    }
    
    .lottery-card p {
        margin: 0.3rem 0;
        font-size: 0.9rem;
        opacity: 0.95;
    }
    
    .status-open {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .status-closed {
        background: linear-gradient(135deg, #636363 0%, #a2ab58 100%);
    }
    
    .status-filled {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    .main-header h1 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
    }
    
    .main-header p {
        color: #eaeaea;
        margin: 0.5rem 0 0 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #666;
    }
    
    /* Filter section */
    .filter-section {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Map container */
    .map-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_lottery_data() -> pd.DataFrame:
    """
    Fetch housing lottery data from NYC Open Data Socrata API
    """
    # Socrata API endpoint for NYC Housing Lottery data
    api_url = "https://data.cityofnewyork.us/resource/vy5i-a666.json"
    
    # Request parameters - get all records with $limit
    params = {
        "$limit": 5000,
        "$order": "lottery_end_date DESC"
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert date columns
        date_columns = ['lottery_start_date', 'lottery_end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['latitude', 'longitude', 'unit_count', 'building_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()


def filter_data(
    df: pd.DataFrame,
    borough: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Filter lottery data based on user selections
    """
    filtered = df.copy()
    
    if borough and borough != "All Boroughs":
        filtered = filtered[filtered['borough'].str.upper() == borough.upper()]
    
    if status and status != "All Statuses":
        filtered = filtered[filtered['lottery_status'].str.contains(status, case=False, na=False)]
    
    if start_date:
        filtered = filtered[filtered['lottery_end_date'] >= pd.Timestamp(start_date)]
    
    if end_date:
        filtered = filtered[filtered['lottery_start_date'] <= pd.Timestamp(end_date)]
    
    return filtered


def get_status_class(status: str) -> str:
    """Return CSS class based on lottery status"""
    if status and 'open' in status.lower():
        return 'status-open'
    elif status and 'filled' in status.lower():
        return 'status-filled'
    return 'status-closed'


def create_map(df: pd.DataFrame) -> folium.Map:
    """
    Create a Folium map with lottery locations
    """
    # Default to NYC center
    center_lat = 40.7128
    center_lon = -74.0060
    
    # Filter rows with valid coordinates
    map_df = df.dropna(subset=['latitude', 'longitude'])
    
    if not map_df.empty:
        center_lat = map_df['latitude'].mean()
        center_lon = map_df['longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='cartodbpositron'
    )
    
    # Add markers for each lottery
    for _, row in map_df.iterrows():
        # Determine marker color based on status
        status = str(row.get('lottery_status', '')).lower()
        if 'open' in status:
            color = 'green'
            icon = 'home'
        elif 'filled' in status:
            color = 'red'
            icon = 'home'
        else:
            color = 'blue'
            icon = 'home'
        
        # Create popup content
        popup_html = f"""
        <div style="width: 250px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{row.get('lottery_name', 'N/A')}</h4>
            <p style="margin: 5px 0;"><strong>Status:</strong> {row.get('lottery_status', 'N/A')}</p>
            <p style="margin: 5px 0;"><strong>Borough:</strong> {row.get('borough', 'N/A')}</p>
            <p style="margin: 5px 0;"><strong>Units:</strong> {row.get('unit_count', 'N/A')}</p>
            <p style="margin: 5px 0;"><strong>Type:</strong> {row.get('development_type', 'N/A')}</p>
            <p style="margin: 5px 0;"><strong>Start:</strong> {pd.to_datetime(row.get('lottery_start_date')).strftime('%m/%d/%Y') if pd.notna(row.get('lottery_start_date')) else 'N/A'}</p>
            <p style="margin: 5px 0;"><strong>End:</strong> {pd.to_datetime(row.get('lottery_end_date')).strftime('%m/%d/%Y') if pd.notna(row.get('lottery_end_date')) else 'N/A'}</p>
        </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix='fa'),
            tooltip=row.get('lottery_name', 'Housing Lottery')
        ).add_to(m)
    
    return m


def display_lottery_card(row: pd.Series):
    """Display a single lottery as a styled card"""
    status_class = get_status_class(row.get('lottery_status', ''))
    
    start_date = pd.to_datetime(row.get('lottery_start_date'))
    end_date = pd.to_datetime(row.get('lottery_end_date'))
    
    start_str = start_date.strftime('%b %d, %Y') if pd.notna(start_date) else 'TBD'
    end_str = end_date.strftime('%b %d, %Y') if pd.notna(end_date) else 'TBD'
    
    st.markdown(f"""
    <div class="lottery-card {status_class}">
        <h3>🏠 {row.get('lottery_name', 'N/A')}</h3>
        <p><strong>📍 Borough:</strong> {row.get('borough', 'N/A')}</p>
        <p><strong>📋 Status:</strong> {row.get('lottery_status', 'N/A')}</p>
        <p><strong>🏗️ Type:</strong> {row.get('development_type', 'N/A')}</p>
        <p><strong>🏢 Units:</strong> {row.get('unit_count', 'N/A')} units in {row.get('building_count', 'N/A')} building(s)</p>
        <p><strong>📅 Application Period:</strong> {start_str} - {end_str}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 NYC Housing Lottery Finder</h1>
        <p>Find and explore affordable housing opportunities across New York City</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner("Loading lottery data..."):
        df = fetch_lottery_data()
    
    if df.empty:
        st.error("Unable to load lottery data. Please try again later.")
        return
    
    # Filters section
    st.markdown("### 🔍 Filter Lotteries")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        boroughs = ["All Boroughs"] + sorted(df['borough'].dropna().unique().tolist())
        selected_borough = st.selectbox("Borough", boroughs, key="borough_filter")
    
    with col2:
        statuses = ["All Statuses", "Open", "Closed", "Filled"]
        selected_status = st.selectbox("Status", statuses, key="status_filter")
    
    with col3:
        min_date = df['lottery_start_date'].min()
        if pd.isna(min_date):
            min_date = datetime.now() - timedelta(days=365)
        start_date = st.date_input(
            "From Date",
            value=datetime.now() - timedelta(days=30),
            key="start_date"
        )
    
    with col4:
        end_date = st.date_input(
            "To Date",
            value=datetime.now() + timedelta(days=180),
            key="end_date"
        )
    
    # Apply filters
    filtered_df = filter_data(
        df,
        borough=selected_borough,
        status=selected_status,
        start_date=start_date,
        end_date=end_date
    )
    
    # Statistics
    st.markdown("### 📊 Summary Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(filtered_df)}</div>
            <div class="stat-label">Total Lotteries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col2:
        open_count = len(filtered_df[filtered_df['lottery_status'].str.contains('Open', case=False, na=False)])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{open_count}</div>
            <div class="stat-label">Currently Open</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col3:
        total_units = filtered_df['unit_count'].sum()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{int(total_units) if pd.notna(total_units) else 0:,}</div>
            <div class="stat-label">Total Units</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col4:
        unique_boroughs = filtered_df['borough'].nunique()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{unique_boroughs}</div>
            <div class="stat-label">Boroughs</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content: Map and List
    tab1, tab2 = st.tabs(["🗺️ Map View", "📋 List View"])
    
    with tab1:
        st.markdown("### Housing Lottery Locations")
        st.caption("Click on markers for details. Green = Open, Red = Filled, Blue = Other")
        
        if not filtered_df.empty:
            lottery_map = create_map(filtered_df)
            st_folium(lottery_map, width=None, height=500, use_container_width=True)
        else:
            st.info("No lotteries found matching your criteria.")
    
    with tab2:
        st.markdown("### Housing Lottery Calendar")
        
        if not filtered_df.empty:
            # Sort by end date
            sorted_df = filtered_df.sort_values('lottery_end_date', ascending=True)
            
            # Option to show only open lotteries first
            show_open_first = st.checkbox("Show open lotteries first", value=True)
            
            if show_open_first:
                open_lotteries = sorted_df[sorted_df['lottery_status'].str.contains('Open', case=False, na=False)]
                other_lotteries = sorted_df[~sorted_df['lottery_status'].str.contains('Open', case=False, na=False)]
                sorted_df = pd.concat([open_lotteries, other_lotteries])
            
            # Pagination
            items_per_page = 10
            total_pages = max(1, (len(sorted_df) - 1) // items_per_page + 1)
            
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="page_number"
            )
            
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            
            page_df = sorted_df.iloc[start_idx:end_idx]
            
            for _, row in page_df.iterrows():
                display_lottery_card(row)
            
            st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(sorted_df))} of {len(sorted_df)} lotteries")
        else:
            st.info("No lotteries found matching your criteria.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
        <p>Data source: <a href="https://data.cityofnewyork.us/Housing-Development/Advertised-Lotteries-on-Housing-Connect-By-Lottery/vy5i-a666" target="_blank">NYC Open Data - Housing Connect Lotteries</a></p>
        <p>For official applications, visit <a href="https://housingconnect.nyc.gov" target="_blank">NYC Housing Connect</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
