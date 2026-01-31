# 🏠 NYC Housing Lottery Finder

A mobile-friendly Streamlit application for exploring affordable housing lottery opportunities in New York City. Browse current and past housing lotteries, view them on an interactive map, and filter by borough, status, and date.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- 🗺️ **Interactive Map View**: Visualize lottery locations across NYC with color-coded markers
- 📋 **Calendar List View**: Browse lotteries in a card-based format sorted by date
- 🔍 **Advanced Filtering**: Filter by borough, lottery status, and date range
- 📊 **Summary Statistics**: Quick overview of total lotteries, open opportunities, and units
- 📱 **Mobile-Friendly**: Responsive design optimized for all devices
- 🔄 **Real-Time Data**: Pulls directly from NYC Open Data API

## Screenshots

### Map View
The interactive map shows all lottery locations with color-coded markers:
- 🟢 Green: Currently Open
- 🔴 Red: All Units Filled
- 🔵 Blue: Other Status

### List View
Browse lotteries in an easy-to-read card format with key details including:
- Development name
- Borough location
- Application period dates
- Unit count
- Status

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/NYCHousingLottery.git
   cd NYCHousingLottery
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run NYCHousingLottery.py
   ```

5. **Open your browser** to `http://localhost:8501`

## Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to `NYCHousingLottery.py`
6. Click "Deploy"

### Deploy to Heroku

1. Create a `Procfile`:
   ```
   web: streamlit run NYCHousingLottery.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create a `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. Deploy to Heroku:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Configuration

### Google Maps API (Optional Enhancement)

While this app uses Folium (OpenStreetMap-based) by default, you can enhance it with Google Maps:

1. Get a Google Maps API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Create a `.streamlit/secrets.toml` file:
   ```toml
   GOOGLE_MAPS_API_KEY = "your-api-key-here"
   ```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_MAPS_API_KEY` | Google Maps JavaScript API key | No |

## Data Source

This application uses the [NYC Open Data - Advertised Lotteries on Housing Connect By Lottery](https://data.cityofnewyork.us/Housing-Development/Advertised-Lotteries-on-Housing-Connect-By-Lottery/vy5i-a666) dataset.

### API Endpoint
```
https://data.cityofnewyork.us/resource/vy5i-a666.json
```

### Key Data Fields
- `lottery_id`: Unique identifier
- `lottery_name`: Development name
- `lottery_status`: Current status (Open, Closed, All Units Filled)
- `development_type`: Rental or Homeownership
- `lottery_start_date`: Application start date
- `lottery_end_date`: Application deadline
- `borough`: NYC borough
- `latitude`/`longitude`: Geographic coordinates
- `unit_count`: Number of available units

## Project Structure

```
NYCHousingLottery/
├── NYCHousingLottery.py    # Main application file
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NYC Open Data](https://opendata.cityofnewyork.us/) for providing the housing lottery data
- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [Folium](https://python-visualization.github.io/folium/) for interactive mapping
- [NYC Housing Connect](https://housingconnect.nyc.gov/) for the official lottery application portal

## Disclaimer

This application is for informational purposes only. For official housing lottery applications, please visit [NYC Housing Connect](https://housingconnect.nyc.gov/). The data displayed is sourced from NYC Open Data and may not reflect real-time availability.

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/yourusername/NYCHousingLottery/issues) on GitHub.
