#!/usr/bin/env python3
"""
🚀 NAS100 Trading Assistant (Stable Release 2.0)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# ======================
# CORE TRADING ENGINE
# ======================

class TradingAssistant:
    def __init__(self):
        self.conn = sqlite3.connect('trading_db.sqlite', check_same_thread=False)
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema"""
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS positions
                              (timestamp DATETIME, asset TEXT, 
                               quantity REAL, entry_price REAL)''')

    def detect_levels(self, df):
        """Robust support/resistance detection with error handling"""
        try:
            # Calculate price range
            df['Range'] = (df['High'] - df['Low']) * 0.5 + df['Low']
            
            # Create numerical bins with explicit dtype
            bins = pd.cut(df['Range'].astype(float), bins=50, include_lowest=True)
            
            # Calculate volume profile with proper interval handling
            vol_profile = df.groupby(bins)['Volume'].sum().reset_index()
            vol_profile['mid'] = vol_profile['Range'].apply(
                lambda x: x.mid if isinstance(x, pd.Interval) else np.nan
            ).astype(float)
            
            # Filter valid levels with non-zero values
            valid_profile = vol_profile.dropna().query('mid > 0')
            
            if valid_profile.empty:
                return [], []
                
            support = valid_profile.nlargest(3, 'Volume')['mid'].values
            resistance = valid_profile.nsmallest(3, 'Volume')['mid'].values
            
            return sorted(support), sorted(resistance)
        
        except Exception as e:
            st.error(f"Technical analysis error: {str(e)}")
            return [], []

# ======================
# STREAMLIT INTERFACE
# ======================

def main():
    st.set_page_config(
        page_title="NAS100 Trading Terminal",
        layout="wide",
        page_icon="💹"
    )
    
    st.title("💹 NAS100 Professional Trading Terminal")
    assistant = TradingAssistant()
    
    # ======================
    # DATA UPLOAD & PROCESSING
    # ======================
    with st.expander("📂 Data Management Hub", expanded=True):
        uploaded_file = st.file_uploader("Import Historical Data", 
                                       type=["csv"],
                                       help="CSV with DateTime, Open, High, Low, Close, Volume")
        
        if uploaded_file:
            try:
                # Load and validate data
                df = pd.read_csv(uploaded_file)
                
                # Clean column names
                df.columns = df.columns.str.strip().str.lower()
                
                # Find datetime column
                dt_col = next((c for c in df.columns if 'date' in c), None)
                
                if dt_col:
                    # Convert and validate datetime
                    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
                    df = df.dropna(subset=[dt_col])
                    
                    if df.empty:
                        st.error("Invalid datetime values in CSV")
                        return
                        
                    # Set index and validate columns
                    df = df.set_index(dt_col)
                    required_cols = {'open', 'high', 'low', 'close', 'volume'}
                    
                    if required_cols.issubset(df.columns):
                        df = df[list(required_cols)].ffill().sort_index()
                        st.session_state.df = df
                        st.success("✅ Data loaded successfully!")
                    else:
                        missing = required_cols - set(df.columns)
                        st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.error("⛔ DateTime column not found. Required format:")
                    st.code("Column name must contain 'date' (case-insensitive)")
                    
            except Exception as e:
                st.error(f"Data processing error: {str(e)}")
                st.markdown("""
                **Required CSV Format:**
                ```csv
                DateTime,Open,High,Low,Close,Volume
                2024-01-01 09:30:00,18000.0,18050.0,17950.0,18000.0,5000
                ```
                """)

    # ======================
    # MARKET ANALYSIS
    # ======================
    if 'df' in st.session_state:
        df = st.session_state.df
        
        with st.container():
            st.header("Technical Analysis Dashboard")
            
            # Detect support/resistance levels
            support, resistance = assistant.detect_levels(df)
            
            # Display levels in columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🛑 Key Support Levels")
                if len(support) > 0:
                    st.dataframe(pd.Series(support, name='Price')
                               .to_frame().style.format("{:.2f}"), 
                               height=200)
                else:
                    st.warning("No support levels detected")
                
            with col2:
                st.subheader("🚀 Key Resistance Levels")
                if len(resistance) > 0:
                    st.dataframe(pd.Series(resistance, name='Price')
                               .to_frame().style.format("{:.2f}"), 
                               height=200)
                else:
                    st.warning("No resistance levels detected")
            
            # Price visualization
            st.subheader("Price Chart")
            st.line_chart(df['close'], use_container_width=True)

    # ======================
    # SAMPLE DATA SYSTEM
    # ======================
    with st.expander("📥 Get Sample Data"):
        sample = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='15T'),
            'open': np.round(np.linspace(18000, 18200, 100), 2),
            'high': np.round(np.linspace(18050, 18250, 100), 2),
            'low': np.round(np.linspace(17950, 18150, 100), 2),
            'close': np.round(np.linspace(18000, 18200, 100), 2),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        st.download_button(
            "⬇️ Download Working Sample",
            sample.to_csv(index=False),
            "nas100_sample_data.csv",
            help="Guaranteed working sample dataset"
        )

if __name__ == "__main__":
    main()
