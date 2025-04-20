#!/usr/bin/env python3
"""
🚀 NAS100 Pro Trading Assistant (Final Certified Version)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import smtplib
import requests
from datetime import datetime
from email.message import EmailMessage

# ======================
# CORE TRADING ENGINE
# ======================

class TradingAssistant:
    def __init__(self):
        self.conn = sqlite3.connect('trading_db.sqlite')
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema"""
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS positions
                              (timestamp DATETIME, asset TEXT, 
                               quantity REAL, entry_price REAL)''')
            self.conn.execute('''CREATE TABLE IF NOT EXISTS audit_log
                              (timestamp DATETIME, action TEXT, details TEXT)''')

    def detect_levels(self, df):
        """Advanced support/resistance detection"""
        try:
            df['Range'] = (df['High'] - df['Low']) * 0.5 + df['Low']
            bins = pd.cut(df['Range'], bins=50, include_lowest=True)
            
            vol_profile = df.groupby(bins)['Volume'].sum().reset_index()
            vol_profile['mid'] = vol_profile['Range'].apply(
                lambda x: x.mid if isinstance(x, pd.Interval) else np.nan
            ).astype(float)
            
            valid_profile = vol_profile.dropna()
            support = valid_profile.nlargest(3, 'Volume')['mid'].values
            resistance = valid_profile.nsmallest(3, 'Volume')['mid'].values
            
            return sorted(support), sorted(resistance)
        
        except Exception as e:
            st.error(f"Technical analysis error: {str(e)}")
            return [], []

    def fetch_live_data(self):
        """Secure market data feed"""
        try:
            response = requests.get(
                f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
                f"&symbol=NDX&interval=5min&apikey={st.secrets['alpha_vantage']['api_key']}"
            )
            response.raise_for_status()
            
            time_series = response.json().get("Time Series (5min)", {})
            df = pd.DataFrame(time_series).T.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }).astype(float)
            
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
            
        except Exception as e:
            st.error(f"Market data error: {str(e)}")
            return None

    def send_alert(self, receiver, message):
        """Secure email notifications"""
        try:
            msg = EmailMessage()
            msg.set_content(f"NAS100 Alert:\n{message}")
            msg['Subject'] = "🚨 Trading Signal Notification"
            msg['From'] = st.secrets["email"]["sender"]
            msg['To'] = receiver
            
            with smtplib.SMTP_SSL(st.secrets["email"]["smtp_server"], 
                                st.secrets["email"]["port"]) as server:
                server.login(st.secrets["email"]["sender"], 
                            st.secrets["email"]["password"])
                server.send_message(msg)
            st.toast("Notification sent successfully!")
        except Exception as e:
            st.error(f"Alert system error: {str(e)}")

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
    # DATA MANAGEMENT
    # ======================
    with st.expander("📂 Data Management Hub", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Import Historical Data", 
                                           type=["csv"],
                                           help="CSV with DateTime, OHLC, Volume")
            
        with col2:
            if st.button("🔄 Sync Live Market Data", 
                        help="Real-time NAS100 prices"):
                with st.spinner("Connecting to market feed..."):
                    live_data = assistant.fetch_live_data()
                    if live_data is not None:
                        st.session_state.df = live_data.reset_index()
                        st.success("Market data synchronized!")

    # ======================
    # DATA PROCESSING
    # ======================
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            dt_col = next((c for c in df.columns if 'date' in c.lower()), None)
            
            if dt_col:
                # Fixed parentheses syntax
                df = df.set_index(pd.to_datetime(df[dt_col]))[['Open', 'High', 'Low', 'Close', 'Volume']]
                st.session_state.df = df.ffill()
            else:
                st.error("⛔ DateTime column not found")
                
        except Exception as e:
            st.error(f"Data processing error: {str(e)}")
            st.markdown("""
            **Required Format:**
            ```csv
            DateTime,Open,High,Low,Close,Volume
            2024-01-01 09:30:00,18000.0,18050.0,17950.0,18000.0,5000
            ```
            """)
    
    elif 'df' in st.session_state:
        df = st.session_state.df

    # ======================
    # TRADING INTERFACE
    # ======================
    if df is not None:
        with st.container():
            st.header("Market Analysis Dashboard")
            
            # Technical Analysis
            support, resistance = assistant.detect_levels(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🛑 Support Levels")
                st.dataframe(pd.Series(support, name='Price')
                           .to_frame().style.format("{:.2f}"), 
                           height=200)
                
            with col2:
                st.subheader("🚀 Resistance Levels")
                st.dataframe(pd.Series(resistance, name='Price')
                           .to_frame().style.format("{:.2f}"), 
                           height=200)
            
            # Visualization
            st.subheader("Price Action")
            st.line_chart(df[['Close']], use_container_width=True)
            
            # Alert System
            with st.expander("🔔 Configure Alerts"):
                alert_email = st.text_input("Notification Email")
                if st.button("💌 Set Price Alerts"):
                    assistant.send_alert(alert_email,
                        f"New levels detected:\nSupport: {support}\nResistance: {resistance}")
            
            # Portfolio Management
            with st.expander("💰 Portfolio Manager"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Current Positions")
                    positions = pd.read_sql("SELECT * FROM positions", assistant.conn)
                    st.dataframe(positions.style.format({
                        'timestamp': lambda x: x.strftime("%Y-%m-%d %H:%M"),
                        'entry_price': "{:.2f}"
                    }))
                    
                with col2:
                    st.subheader("Execute Trade")
                    trade_qty = st.number_input("Shares", 1, 1000, 100)
                    trade_price = st.number_input("Price", 
                                                value=float(df['Close'].iloc[-1]))
                    
                    if st.button("📈 Buy NAS100"):
                        assistant.conn.execute('''
                            INSERT INTO positions VALUES (?,?,?,?)
                        ''', (datetime.now(), "NAS100", trade_qty, trade_price))
                        assistant.conn.commit()
                        st.success("Trade executed successfully!")

    # ======================
    # SAMPLE DATA SYSTEM
    # ======================
    with st.expander("📥 Get Starter Data"):
        sample = pd.DataFrame({
            'DateTime': pd.date_range('2024-01-01', periods=100, freq='15T'),
            'Open': np.round(np.linspace(18000, 18200, 100), 2),
            'High': np.round(np.linspace(18050, 18250, 100), 2),
            'Low': np.round(np.linspace(17950, 18150, 100), 2),
            'Close': np.round(np.linspace(18000, 18200, 100), 2),
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
        st.download_button(
            "⬇️ Download Verified Sample",
            sample.to_csv(index=False),
            "nas100_training_data.csv",
            help="Perfectly formatted sample dataset"
        )

if __name__ == "__main__":
    main()
