#!/usr/bin/env python3
"""
NAS100 AI TRADING ASSISTANT - WITH IMPROVED DATA FETCHING

INSTRUCTIONS:
1. For Alpha Vantage API, get a free key from https://www.alphavantage.co/
2. For Twelve Data API, get a free key from https://twelvedata.com/
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
from io import StringIO
from datetime import datetime, timedelta

# ======================
# DATA FETCHING FUNCTIONS
# ======================

# API Configuration
API_KEYS = {
    'alpha_vantage': 'YOUR_ALPHA_VANTAGE_API_KEY',  # Replace with your actual key
    'twelve_data': 'YOUR_TWELVE_DATA_API_KEY'       # Replace with your actual key
}

def fetch_realtime_nas100():
    """Fetch real-time NAS100 price from Twelve Data API"""
    try:
        url = f"https://api.twelvedata.com/price?symbol=NDX&apikey={API_KEYS['twelve_data']}"
        response = requests.get(url)
        data = response.json()
        
        if 'price' in data:
            price = float(data['price'])
            return {
                'price': price,
                'change': 0,  # Real-time APIs often don't provide change in basic tier
                'change_pct': 0
            }
        return None
    except Exception as e:
        st.error(f"Error fetching real-time data: {str(e)}")
        return None

def fetch_historical_nas100(period='1mo'):
    """Fetch historical NAS100 data with fallback to different APIs"""
    try:
        # First try Alpha Vantage
        df = fetch_alpha_vantage(period)
        if df is not None:
            return df
            
        # If Alpha Vantage fails, try Twelve Data
        st.warning("Alpha Vantage failed, trying Twelve Data...")
        df = fetch_twelve_data(period)
        return df
        
    except Exception as e:
        st.error(f"All data sources failed: {str(e)}")
        return None

def fetch_alpha_vantage(period):
    """Fetch from Alpha Vantage API"""
    try:
        period_map = {
            '1d': 1,
            '1w': 7,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365
        }
        
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NDX&interval=5min&outputsize=full&apikey={API_KEYS['alpha_vantage']}"
        
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series (5min)' not in data:
            st.error(f"Alpha Vantage Error: {data.get('Note', 'Unknown error')}")
            return None
            
        raw_data = data['Time Series (5min)']
        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert and rename columns
        df = df.astype(float)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Filter for selected period
        days = period_map.get(period, 30)
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff_date]
        
        return df
    except Exception as e:
        st.error(f"Alpha Vantage fetch failed: {str(e)}")
        return None

def fetch_twelve_data(period):
    """Fetch from Twelve Data API"""
    try:
        period_map = {
            '1d': '1day',
            '1w': '1week',
            '1mo': '1month',
            '3mo': '3month',
            '6mo': '6month',
            '1y': '1year'
        }
        
        interval = '5min' if period in ['1d', '1w'] else '1day'
        url = f"https://api.twelvedata.com/time_series?symbol=NDX&interval={interval}&outputsize=5000&apikey={API_KEYS['twelve_data']}"
        
        response = requests.get(url)
        data = response.json()
        
        if 'values' not in data:
            st.error(f"Twelve Data Error: {data.get('message', 'Unknown error')}")
            return None
            
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').set_index('datetime')
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return df
    except Exception as e:
        st.error(f"Twelve Data fetch failed: {str(e)}")
        return None

# ======================
# CORE TRADING FUNCTIONS
# ======================

# ... [rest of your existing trading functions remain unchanged] ...

def main():
    st.set_page_config(
        page_title="NAS100 Trading Assistant",
        layout="wide",
        page_icon="📈"
    )
    
    st.title("📊 NAS100 AI Trading Assistant")
    
    # API Key Configuration
    with st.expander("API Configuration (Required for live data)"):
        API_KEYS['alpha_vantage'] = st.text_input("Alpha Vantage API Key", API_KEYS['alpha_vantage'])
        API_KEYS['twelve_data'] = st.text_input("Twelve Data API Key", API_KEYS['twelve_data'])
    
    # Real-time price display
    st.sidebar.subheader("Real-time NAS100 Price")
    if st.sidebar.button("Refresh Price"):
        price_data = fetch_realtime_nas100()
        if price_data:
            st.sidebar.metric("NAS100 Price", 
                            f"{price_data['price']:.2f}",
                            f"{price_data['change']:.2f} ({price_data['change_pct']:.2f}%)")
    
    # Data source selection
    data_source = st.radio("Select Data Source:", 
                          ["Upload CSV", "Fetch Historical Data"])
    
    if data_source == "Fetch Historical Data":
        st.subheader("Fetch Historical NAS100 Data")
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Time Period", 
                                ['1d', '1w', '1mo', '3mo', '6mo', '1y'],
                                index=2)
        with col2:
            if st.button("Fetch Data"):
                with st.spinner("Downloading historical data..."):
                    df = fetch_historical_nas100(period)
                    if df is not None:
                        st.session_state.df = df
                        st.success(f"Successfully fetched {len(df)} data points")
                        st.write(df.head())  # Show preview of fetched data
    else:
        # File upload
        uploaded_file = st.file_uploader("Upload NAS100 Data (CSV)", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['Datetime'], index_col='Datetime')
                st.session_state.df = df
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # ... [rest of your existing main function remains unchanged] ...

if __name__ == "__main__":
    main()
