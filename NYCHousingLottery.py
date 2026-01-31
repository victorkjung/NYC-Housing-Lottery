"""
NYC Housing Lottery Finder
A Streamlit app to browse, map, and analyze NYC affordable housing lotteries
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="NYC Housing Lottery Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Column definitions with descriptions
COLUMN_DEFINITIONS = {
    "lottery_id": {
        "display_name": "Lottery ID",
        "description": "Unique identifier for the housing lottery"
    },
    "lottery_name": {
        "display_name": "Lottery Name",
        "description": "Name of the housing development/lottery"
    },
    "lottery_status": {
        "display_name": "Lottery Status",
        "description": "Current status: Open, Closed, or All Units Filled"
    },
    "development_type": {
        "display_name": "Development Type",
        "description": "Type of housing: Rental or Homeownership"
    },
    "lottery_start_date": {
        "display_name": "Start Date",
        "description": "Date when lottery applications open"
    },
    "lottery_end_date": {
        "display_name": "End Date",
        "description": "Application deadline for the lottery"
    },
    "building_count": {
        "display_name": "Building Count",
        "description": "Number of buildings in the development"
    },
    "unit_count": {
        "display_name": "Total Units",
        "description": "Total number of available housing units"
    },
    "unit_distribution_studio": {
        "display_name": "Studio Units",
        "description": "Number of studio apartments available"
    },
    "unit_distribution_1_bedroom": {
        "display_name": "1-Bedroom Units",
        "description": "Number of 1-bedroom apartments available"
    },
    "unit_distribution_2_bedrooms": {
        "display_name": "2-Bedroom Units",
        "description": "Number of 2-bedroom apartments available"
    },
    "unit_distribution_3_bedrooms": {
        "display_name": "3-Bedroom Units",
        "description": "Number of 3-bedroom apartments available"
    },
    "unit_distribution_4_bedroom": {
        "display_name": "4+ Bedroom Units",
        "description": "Number of 4 or more bedroom apartments available"
    },
    "applied_income_ami_category_extremely_low_income": {
        "display_name": "Extremely Low Income",
        "description": "Units for households at 0-30% Area Median Income"
    },
    "applied_income_ami_category_very_low_income": {
        "display_name": "Very Low Income",
        "description": "Units for households at 31-50% Area Median Income"
    },
    "applied_income_ami_category_low_income": {
        "display_name": "Low Income",
        "description": "Units for households at 51-80% Area Median Income"
    },
    "applied_income_ami_category_moderate_income": {
        "display_name": "Moderate Income",
        "description": "Units for households at 81-120% Area Median Income"
    },
    "applied_income_ami_category_middle_income": {
        "display_name": "Middle Income",
        "description": "Units for households at 121-165% Area Median Income"
    },
    "applied_income_ami_category_above_middle_income": {
        "display_name": "Above Middle Income",
        "description": "Units for households above 165% Area Median Income"
    },
    "lottery_mobility_percentage": {
        "display_name": "Mobility %",
        "description": "Percentage of units for applicants with mobility disabilities"
    },
    "lottery_vision_hearing_percentage": {
        "display_name": "Vision/Hearing %",
        "description": "Percentage of units for applicants with vision/hearing disabilities"
    },
    "lottery_community_board_percentage": {
        "display_name": "Community Board %",
        "description": "Percentage of units reserved for community board residents"
    },
    "lottery_municipal_employee_military_veteran_percentage": {
        "display_name": "Municipal/Veteran %",
        "description": "Percentage for municipal employees and military veterans"
    },
    "lottery_nycha_percentage": {
        "display_name": "NYCHA %",
        "description": "Percentage of units for NYCHA residents"
    },
    "lottery_senior_percentage": {
        "display_name": "Senior %",
        "description": "Percentage of units reserved for seniors"
    },
    "borough": {
        "display_name": "Borough",
        "description": "NYC borough where the development is located"
    },
    "postcode": {
        "display_name": "Zip Code",
        "description": "Postal code of the development"
    },
    "community_board": {
        "display_name": "Community Board",
        "description": "NYC Community Board district number"
    },
    "latitude": {
        "display_name": "Latitude",
        "description": "Geographic latitude coordinate"
    },
    "longitude": {
        "display_name": "Longitude",
        "description": "Geographic longitude coordinate"
    }
}

# Unit distribution columns
UNIT_DIST_COLS = [
    "unit_distribution_studio",
    "unit_distribution_1_bedroom",
    "unit_distribution_2_bedrooms",
    "unit_distribution_3_bedrooms",
    "unit_distribution_4_bedroom"
]

# AMI category columns
AMI_COLS = [
    "applied_income_ami_category_extremely_low_income",
    "applied_income_ami_category_very_low_income",
    "applied_income_ami_category_low_income",
    "applied_income_ami_category_moderate_income",
    "applied_income_ami_category_middle_income",
    "applied_income_ami_category_above_middle_income"
]

# Lottery preference percentage columns
LOTTERY_PCT_COLS = [
    "lottery_mobility_percentage",
    "lottery_vision_hearing_percentage",
    "lottery_community_board_percentage",
    "lottery_municipal_employee_military_veteran_percentage",
    "lottery_nycha_percentage",
    "lottery_senior_percentage"
]

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
    
    /* Column description tooltip */
    .col-desc {
        font-size: 0.75rem;
        color: #666;
        font-style: italic;
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
    
    /* Download button styling */
    .download-btn {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_lottery_data() -> pd.DataFrame:
    """
    Fetch housing lottery data from NYC Open Data Socrata API
    """
    api_url = "https://data.cityofnewyork.us/resource/vy5i-a666.json"
    
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
        
        # Convert all numeric columns
        numeric_columns = [
            'latitude', 'longitude', 'unit_count', 'building_count',
            'unit_distribution_studio', 'unit_distribution_1_bedroom',
            'unit_distribution_2_bedrooms', 'unit_distribution_3_bedrooms',
            'unit_distribution_4_bedroom',
            'applied_income_ami_category_extremely_low_income',
            'applied_income_ami_category_very_low_income',
            'applied_income_ami_category_low_income',
            'applied_income_ami_category_moderate_income',
            'applied_income_ami_category_middle_income',
            'applied_income_ami_category_above_middle_income',
            'lottery_mobility_percentage', 'lottery_vision_hearing_percentage',
            'lottery_community_board_percentage',
            'lottery_municipal_employee_military_veteran_percentage',
            'lottery_nycha_percentage', 'lottery_senior_percentage'
        ]
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
    end_date: Optional[datetime] = None,
    development_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter lottery data based on user selections
    """
    filtered = df.copy()
    
    if borough and borough != "All Boroughs" and 'borough' in filtered.columns:
        filtered = filtered[filtered['borough'].str.upper() == borough.upper()]
    
    if status and status != "All Statuses" and 'lottery_status' in filtered.columns:
        filtered = filtered[filtered['lottery_status'].str.contains(status, case=False, na=False)]
    
    if development_type and development_type != "All Types" and 'development_type' in filtered.columns:
        filtered = filtered[filtered['development_type'].str.contains(development_type, case=False, na=False)]
    
    if start_date and 'lottery_end_date' in filtered.columns:
        filtered = filtered[filtered['lottery_end_date'] >= pd.Timestamp(start_date)]
    
    if end_date and 'lottery_start_date' in filtered.columns:
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
    center_lat = 40.7128
    center_lon = -74.0060
    
    map_df = df.dropna(subset=['latitude', 'longitude'])
    
    if not map_df.empty:
        center_lat = map_df['latitude'].mean()
        center_lon = map_df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='cartodbpositron'
    )
    
    for _, row in map_df.iterrows():
        status = str(row.get('lottery_status', '')).lower()
        if 'open' in status:
            color = 'green'
        elif 'filled' in status:
            color = 'red'
        else:
            color = 'blue'
        
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
            icon=folium.Icon(color=color, icon='home', prefix='fa'),
            tooltip=row.get('lottery_name', 'Housing Lottery')
        ).add_to(m)
    
    return m


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes for download"""
    return df.to_csv(index=False).encode('utf-8')


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


def display_detailed_lottery_info(row: pd.Series):
    """Display detailed lottery information with all columns"""
    with st.expander(f"🏠 {row.get('lottery_name', 'N/A')} - {row.get('lottery_status', 'N/A')}", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Basic Information")
            for key in ['lottery_id', 'lottery_name', 'lottery_status', 'development_type', 
                       'lottery_start_date', 'lottery_end_date', 'building_count', 'unit_count']:
                if key in COLUMN_DEFINITIONS and key in row.index:
                    value = row.get(key, 'N/A')
                    if pd.isna(value):
                        value = 'N/A'
                    elif 'date' in key and pd.notna(value):
                        value = pd.to_datetime(value).strftime('%m/%d/%Y')
                    st.markdown(f"**{COLUMN_DEFINITIONS[key]['display_name']}:** {value}")
                    st.caption(COLUMN_DEFINITIONS[key]['description'])
        
        with col2:
            st.markdown("#### Unit Distribution")
            for key in UNIT_DIST_COLS:
                if key in COLUMN_DEFINITIONS and key in row.index:
                    value = row.get(key, 0)
                    if pd.isna(value):
                        value = 0
                    st.markdown(f"**{COLUMN_DEFINITIONS[key]['display_name']}:** {int(value)}")
                    st.caption(COLUMN_DEFINITIONS[key]['description'])
            
            st.markdown("#### Location")
            for key in ['borough', 'postcode', 'community_board']:
                if key in COLUMN_DEFINITIONS and key in row.index:
                    value = row.get(key, 'N/A')
                    if pd.isna(value):
                        value = 'N/A'
                    st.markdown(f"**{COLUMN_DEFINITIONS[key]['display_name']}:** {value}")
        
        with col3:
            st.markdown("#### Income Categories (AMI)")
            for key in AMI_COLS:
                if key in COLUMN_DEFINITIONS and key in row.index:
                    value = row.get(key, 0)
                    if pd.isna(value):
                        value = 0
                    st.markdown(f"**{COLUMN_DEFINITIONS[key]['display_name']}:** {int(value)}")
            
            st.markdown("#### Lottery Preferences (%)")
            for key in LOTTERY_PCT_COLS:
                if key in COLUMN_DEFINITIONS and key in row.index:
                    value = row.get(key, 0)
                    if pd.isna(value):
                        value = 0
                    st.markdown(f"**{COLUMN_DEFINITIONS[key]['display_name']}:** {value}%")


def render_unit_distribution_tab(df: pd.DataFrame):
    """Render the Unit Distribution analysis tab"""
    st.markdown("### 🏢 Unit Distribution Analysis")
    st.markdown("Analyze the distribution of unit sizes across housing lotteries.")
    
    # Filters specific to this tab
    st.markdown("#### Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        ud_boroughs = ["All Boroughs"] + sorted(df['borough'].dropna().unique().tolist())
        ud_borough = st.selectbox("Borough", ud_boroughs, key="ud_borough")
    
    with filter_col2:
        ud_statuses = ["All Statuses", "Open", "Closed", "Filled"]
        ud_status = st.selectbox("Status", ud_statuses, key="ud_status")
    
    with filter_col3:
        ud_types = ["All Types"] + sorted(df['development_type'].dropna().unique().tolist())
        ud_type = st.selectbox("Development Type", ud_types, key="ud_type")
    
    # Apply filters
    filtered = filter_data(df, borough=ud_borough, status=ud_status, development_type=ud_type)
    
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Calculate totals
    unit_totals = {}
    for col in UNIT_DIST_COLS:
        if col in filtered.columns:
            total = filtered[col].sum()
            if pd.notna(total):
                unit_totals[COLUMN_DEFINITIONS[col]['display_name']] = total
    
    if not unit_totals or sum(unit_totals.values()) == 0:
        st.info("No unit distribution data available for the selected filters.")
        return
    
    # Create charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Pie chart
        fig_pie = px.pie(
            values=list(unit_totals.values()),
            names=list(unit_totals.keys()),
            title="Unit Distribution by Size",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col2:
        # Bar chart
        fig_bar = px.bar(
            x=list(unit_totals.keys()),
            y=list(unit_totals.values()),
            title="Total Units by Size",
            labels={'x': 'Unit Size', 'y': 'Number of Units'},
            color=list(unit_totals.keys()),
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # By Borough breakdown
    st.markdown("#### Unit Distribution by Borough")
    borough_data = []
    for borough in filtered['borough'].dropna().unique():
        borough_df = filtered[filtered['borough'] == borough]
        row_data = {'Borough': borough}
        for col in UNIT_DIST_COLS:
            if col in borough_df.columns:
                row_data[COLUMN_DEFINITIONS[col]['display_name']] = borough_df[col].sum()
        borough_data.append(row_data)
    
    borough_summary = pd.DataFrame(borough_data)
    if not borough_summary.empty:
        fig_stacked = px.bar(
            borough_summary,
            x='Borough',
            y=[COLUMN_DEFINITIONS[col]['display_name'] for col in UNIT_DIST_COLS if col in filtered.columns],
            title="Unit Distribution by Borough",
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
    
    # Data table and download
    st.markdown("#### Detailed Data")
    display_cols = ['lottery_name', 'borough', 'lottery_status'] + [col for col in UNIT_DIST_COLS if col in filtered.columns]
    display_df = filtered[display_cols].copy()
    display_df.columns = ['Lottery Name', 'Borough', 'Status'] + [COLUMN_DEFINITIONS[col]['display_name'] for col in UNIT_DIST_COLS if col in filtered.columns]
    
    st.dataframe(display_df, use_container_width=True, height=300)
    
    csv_data = convert_df_to_csv(display_df)
    st.download_button(
        label="📥 Download Unit Distribution Data (CSV)",
        data=csv_data,
        file_name=f"unit_distribution_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_unit_dist"
    )


def render_ami_category_tab(df: pd.DataFrame):
    """Render the AMI Category analysis tab"""
    st.markdown("### 💰 Applied Income AMI Category Analysis")
    st.markdown("Analyze units by Area Median Income (AMI) eligibility categories.")
    
    # AMI Category descriptions
    with st.expander("ℹ️ What are AMI Categories?"):
        st.markdown("""
        **Area Median Income (AMI)** is used to determine eligibility for affordable housing:
        - **Extremely Low Income**: 0-30% of AMI
        - **Very Low Income**: 31-50% of AMI
        - **Low Income**: 51-80% of AMI
        - **Moderate Income**: 81-120% of AMI
        - **Middle Income**: 121-165% of AMI
        - **Above Middle Income**: Above 165% of AMI
        """)
    
    # Filters
    st.markdown("#### Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        ami_boroughs = ["All Boroughs"] + sorted(df['borough'].dropna().unique().tolist())
        ami_borough = st.selectbox("Borough", ami_boroughs, key="ami_borough")
    
    with filter_col2:
        ami_statuses = ["All Statuses", "Open", "Closed", "Filled"]
        ami_status = st.selectbox("Status", ami_statuses, key="ami_status")
    
    with filter_col3:
        ami_types = ["All Types"] + sorted(df['development_type'].dropna().unique().tolist())
        ami_type = st.selectbox("Development Type", ami_types, key="ami_type")
    
    # Apply filters
    filtered = filter_data(df, borough=ami_borough, status=ami_status, development_type=ami_type)
    
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Calculate totals
    ami_totals = {}
    for col in AMI_COLS:
        if col in filtered.columns:
            total = filtered[col].sum()
            if pd.notna(total):
                ami_totals[COLUMN_DEFINITIONS[col]['display_name']] = total
    
    # Summary metrics
    if ami_totals:
        st.markdown("#### Summary")
        num_cols = min(len(ami_totals), 6)  # Max 6 columns
        metric_cols = st.columns(num_cols)
        for i, (name, value) in enumerate(ami_totals.items()):
            with metric_cols[i % num_cols]:
                st.metric(name, f"{int(value):,}")
    
    # Charts
    if ami_totals and sum(ami_totals.values()) > 0:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Donut chart
            fig_donut = px.pie(
                values=list(ami_totals.values()),
                names=list(ami_totals.keys()),
                title="Units by AMI Category",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        
        with chart_col2:
            # Horizontal bar chart
            fig_hbar = px.bar(
                y=list(ami_totals.keys()),
                x=list(ami_totals.values()),
                title="Total Units by Income Category",
                labels={'x': 'Number of Units', 'y': 'AMI Category'},
                orientation='h',
                color=list(ami_totals.values()),
                color_continuous_scale='Viridis'
            )
            fig_hbar.update_layout(showlegend=False)
            st.plotly_chart(fig_hbar, use_container_width=True)
    else:
        st.info("No AMI category data available for the selected filters.")
    
    # Trend over time
    st.markdown("#### AMI Distribution Over Time")
    time_data = filtered.copy()
    time_data['year_month'] = time_data['lottery_start_date'].dt.to_period('M').astype(str)
    time_grouped = time_data.groupby('year_month')[AMI_COLS].sum().reset_index()
    time_grouped.columns = ['Period'] + [COLUMN_DEFINITIONS[col]['display_name'] for col in AMI_COLS]
    
    if len(time_grouped) > 1:
        fig_line = px.line(
            time_grouped,
            x='Period',
            y=[COLUMN_DEFINITIONS[col]['display_name'] for col in AMI_COLS],
            title="AMI Category Trends Over Time",
            markers=True
        )
        fig_line.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Data table and download
    st.markdown("#### Detailed Data")
    display_cols = ['lottery_name', 'borough', 'lottery_status'] + [col for col in AMI_COLS if col in filtered.columns]
    display_df = filtered[display_cols].copy()
    display_df.columns = ['Lottery Name', 'Borough', 'Status'] + [COLUMN_DEFINITIONS[col]['display_name'] for col in AMI_COLS if col in filtered.columns]
    
    st.dataframe(display_df, use_container_width=True, height=300)
    
    csv_data = convert_df_to_csv(display_df)
    st.download_button(
        label="📥 Download AMI Category Data (CSV)",
        data=csv_data,
        file_name=f"ami_categories_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_ami"
    )


def render_lottery_percentage_tab(df: pd.DataFrame):
    """Render the Lottery Preference Percentage analysis tab"""
    st.markdown("### 📊 Lottery Preference Percentage Analysis")
    st.markdown("Analyze lottery preference allocations for special populations.")
    
    # Preference descriptions
    with st.expander("ℹ️ What are Lottery Preferences?"):
        st.markdown("""
        NYC Housing Lotteries may reserve percentages of units for:
        - **Mobility**: Applicants with mobility disabilities
        - **Vision/Hearing**: Applicants with vision or hearing disabilities
        - **Community Board**: Residents of the local community board district
        - **Municipal/Veteran**: NYC municipal employees and military veterans
        - **NYCHA**: Current NYCHA (public housing) residents
        - **Senior**: Seniors (typically 62+)
        """)
    
    # Filters
    st.markdown("#### Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        lp_boroughs = ["All Boroughs"] + sorted(df['borough'].dropna().unique().tolist())
        lp_borough = st.selectbox("Borough", lp_boroughs, key="lp_borough")
    
    with filter_col2:
        lp_statuses = ["All Statuses", "Open", "Closed", "Filled"]
        lp_status = st.selectbox("Status", lp_statuses, key="lp_status")
    
    with filter_col3:
        lp_types = ["All Types"] + sorted(df['development_type'].dropna().unique().tolist())
        lp_type = st.selectbox("Development Type", lp_types, key="lp_type")
    
    # Apply filters
    filtered = filter_data(df, borough=lp_borough, status=lp_status, development_type=lp_type)
    
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Calculate averages
    pct_avgs = {}
    for col in LOTTERY_PCT_COLS:
        if col in filtered.columns:
            avg_val = filtered[col].mean()
            if pd.notna(avg_val):
                pct_avgs[COLUMN_DEFINITIONS[col]['display_name']] = avg_val
    
    # Summary metrics
    if pct_avgs:
        st.markdown("#### Average Preference Percentages")
        num_cols = min(len(pct_avgs), 3)  # Max 3 columns per row
        metric_cols = st.columns(num_cols)
        for i, (name, value) in enumerate(pct_avgs.items()):
            with metric_cols[i % num_cols]:
                st.metric(name, f"{value:.1f}%")
    
    # Charts
    if pct_avgs and len(pct_avgs) > 0:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Radar chart
            categories = list(pct_avgs.keys())
            values = list(pct_avgs.values())
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Average %',
                line_color='#667eea'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2 if values else 100])),
                showlegend=False,
                title="Average Preference Percentages"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with chart_col2:
            # Box plot for distribution
            box_data = []
            for col in LOTTERY_PCT_COLS:
                if col in filtered.columns:
                    for val in filtered[col].dropna():
                        box_data.append({
                            'Preference': COLUMN_DEFINITIONS[col]['display_name'],
                            'Percentage': val
                        })
            
            if box_data:
                box_df = pd.DataFrame(box_data)
                fig_box = px.box(
                    box_df,
                    x='Preference',
                    y='Percentage',
                    title="Distribution of Preference Percentages",
                    color='Preference',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_box.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No distribution data available.")
    else:
        st.info("No lottery preference data available for the selected filters.")
    
    # By Borough comparison
    st.markdown("#### Preference Percentages by Borough")
    borough_pct_data = []
    for borough in filtered['borough'].dropna().unique():
        borough_df = filtered[filtered['borough'] == borough]
        row_data = {'Borough': borough}
        for col in LOTTERY_PCT_COLS:
            if col in borough_df.columns:
                row_data[COLUMN_DEFINITIONS[col]['display_name']] = borough_df[col].mean()
        borough_pct_data.append(row_data)
    
    borough_pct_summary = pd.DataFrame(borough_pct_data)
    if not borough_pct_summary.empty and len(borough_pct_summary) > 1:
        fig_grouped = px.bar(
            borough_pct_summary,
            x='Borough',
            y=[COLUMN_DEFINITIONS[col]['display_name'] for col in LOTTERY_PCT_COLS if col in filtered.columns],
            title="Average Preference % by Borough",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_grouped, use_container_width=True)
    
    # Data table and download
    st.markdown("#### Detailed Data")
    display_cols = ['lottery_name', 'borough', 'lottery_status'] + [col for col in LOTTERY_PCT_COLS if col in filtered.columns]
    display_df = filtered[display_cols].copy()
    display_df.columns = ['Lottery Name', 'Borough', 'Status'] + [COLUMN_DEFINITIONS[col]['display_name'] for col in LOTTERY_PCT_COLS if col in filtered.columns]
    
    st.dataframe(display_df, use_container_width=True, height=300)
    
    csv_data = convert_df_to_csv(display_df)
    st.download_button(
        label="📥 Download Lottery Preferences Data (CSV)",
        data=csv_data,
        file_name=f"lottery_preferences_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_lottery_pct"
    )


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
    
    # Global Filters section
    st.markdown("### 🔍 Filter Lotteries")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        boroughs = ["All Boroughs"] + sorted(df['borough'].dropna().unique().tolist())
        selected_borough = st.selectbox("Borough", boroughs, key="main_borough_filter")
    
    with col2:
        statuses = ["All Statuses", "Open", "Closed", "Filled"]
        selected_status = st.selectbox("Status", statuses, key="main_status_filter")
    
    with col3:
        min_date = df['lottery_start_date'].min()
        if pd.isna(min_date):
            min_date = datetime.now() - timedelta(days=365)
        start_date = st.date_input(
            "From Date",
            value=datetime.now() - timedelta(days=30),
            key="main_start_date"
        )
    
    with col4:
        end_date = st.date_input(
            "To Date",
            value=datetime.now() + timedelta(days=180),
            key="main_end_date"
        )
    
    # Apply global filters
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
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Map View", 
        "📋 List View", 
        "🏢 Unit Distribution",
        "💰 AMI Categories",
        "📊 Lottery Preferences"
    ])
    
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
        st.markdown("Click on each lottery to view all details including unit distribution, AMI categories, and preferences.")
        
        if not filtered_df.empty:
            # Sort options
            sort_col1, sort_col2 = st.columns([2, 2])
            with sort_col1:
                show_open_first = st.checkbox("Show open lotteries first", value=True)
            with sort_col2:
                view_mode = st.radio("View Mode", ["Card View", "Detailed View", "Table View"], horizontal=True)
            
            sorted_df = filtered_df.sort_values('lottery_end_date', ascending=True)
            
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
                key="list_page_number"
            )
            
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_df = sorted_df.iloc[start_idx:end_idx]
            
            if view_mode == "Card View":
                for _, row in page_df.iterrows():
                    display_lottery_card(row)
            elif view_mode == "Detailed View":
                for _, row in page_df.iterrows():
                    display_detailed_lottery_info(row)
            else:  # Table View
                # Prepare display columns with friendly names
                table_df = page_df.copy()
                rename_dict = {col: COLUMN_DEFINITIONS[col]['display_name'] 
                              for col in table_df.columns if col in COLUMN_DEFINITIONS}
                table_df = table_df.rename(columns=rename_dict)
                st.dataframe(table_df, use_container_width=True, height=400)
            
            st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(sorted_df))} of {len(sorted_df)} lotteries")
            
            # Download full filtered data
            st.markdown("---")
            full_csv = convert_df_to_csv(filtered_df)
            st.download_button(
                label="📥 Download All Filtered Data (CSV)",
                data=full_csv,
                file_name=f"nyc_housing_lotteries_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_all"
            )
        else:
            st.info("No lotteries found matching your criteria.")
    
    with tab3:
        render_unit_distribution_tab(df)
    
    with tab4:
        render_ami_category_tab(df)
    
    with tab5:
        render_lottery_percentage_tab(df)
    
    # Column Reference
    with st.expander("📖 Column Reference Guide"):
        st.markdown("### Data Field Descriptions")
        ref_data = []
        for col, info in COLUMN_DEFINITIONS.items():
            ref_data.append({
                "Field Name": col,
                "Display Name": info['display_name'],
                "Description": info['description']
            })
        ref_df = pd.DataFrame(ref_data)
        st.dataframe(ref_df, use_container_width=True, height=400)
    
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
