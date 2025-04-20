#!/usr/bin/env python3
"""
NAS100 AI TRADING ASSISTANT - SINGLE FILE SOLUTION WITH REAL-TIME DATA

INSTRUCTIONS:

1. INSTALLATION:
   ----------------------------
   # For Windows:
   python -m venv venv
   venv\Scripts\activate
   pip install streamlit pandas numpy requests

   # For Mac/Linux:
   python3 -m venv venv
   source venv/bin/activate
   pip install streamlit pandas numpy requests

2. RUNNING:
   ----------------------------
   streamlit run NAS100_Trading_Assistant.py

3. TROUBLESHOOTING:
   ----------------------------
   If you get dependency conflicts:
   pip install --force-reinstall -r requirements.txt

   For port conflicts:
   streamlit run --server.port 8502 NAS100_Trading_Assistant.py
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

def fetch_realtime_nas100():
    """Fetch real-time NAS100 price from Alpha Vantage API"""
    try:
        # Using Alpha Vantage API (free tier)
        api_key = 'YOUR_API_KEY'  # Replace with your actual API key
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=NDX&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if 'Global Quote' in data:
            quote = data['Global Quote']
            return {
                'price': float(quote['05. price']),
                'change': float(quote['09. change']),
                'change_pct': float(quote['10. change percent'].rstrip('%'))
            }
        return None
    except Exception as e:
        st.error(f"Error fetching real-time data: {str(e)}")
        return None

def fetch_historical_nas100(period='1mo'):
    """Fetch historical NAS100 data from Alpha Vantage"""
    try:
        # Period to days mapping
        period_map = {
            '1d': 1,
            '1w': 7,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365
        }
        
        api_key = 'YOUR_API_KEY'  # Replace with your actual API key
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NDX&interval=5min&outputsize=full&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series (5min)' not in data:
            st.error("Failed to fetch historical data")
            return None
            
        raw_data = data['Time Series (5min)']
        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert string values to float
        df = df.astype(float)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Filter for the selected period
        days = period_map.get(period, 30)
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff_date]
        
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

# ======================
# CORE TRADING FUNCTIONS
# ======================

def detect_support_resistance(df, window=20):
    """Identify support/resistance levels using swing points"""
    highs = df['High']
    lows = df['Low']
    
    swing_highs = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_lows = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
    
    def cluster_levels(levels):
        clusters = []
        for level in levels:
            found = False
            for cluster in clusters:
                if abs(level - cluster['mean']) < df['Close'].mean() * 0.002:
                    cluster['points'].append(level)
                    cluster['mean'] = np.mean(cluster['points'])
                    found = True
                    break
            if not found:
                clusters.append({'points': [level], 'mean': level})
        return [x['mean'] for x in clusters if len(x['points']) >= 2]
    
    support = cluster_levels(swing_lows)
    resistance = cluster_levels(swing_highs)
    
    return sorted(support), sorted(resistance)

def generate_signals(df, support, resistance, use_volume=True):
    """Generate trading signals at key levels"""
    signals = []
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Support bounce (BUY)
        for level in support:
            if (prev['Low'] <= level * 1.001) and (current['Close'] > level * 1.001):
                if not use_volume or current['Volume'] > df['Volume'].rolling(20).mean().iloc[i]:
                    signals.append({
                        'Datetime': df.index[i],
                        'Signal': 'Buy',
                        'Price': current['Close'],
                        'Type': 'Support Bounce'
                    })
                    break
        
        # Resistance rejection (SELL)
        for level in resistance:
            if (prev['High'] >= level * 0.999) and (current['Close'] < level * 0.999):
                if not use_volume or current['Volume'] > df['Volume'].rolling(20).mean().iloc[i]:
                    signals.append({
                        'Datetime': df.index[i],
                        'Signal': 'Sell', 
                        'Price': current['Close'],
                        'Type': 'Resistance Reject'
                    })
                    break
    
    return pd.DataFrame(signals)

def backtest(df, signals, sl_pct=1.5, tp_pct=3.0):
    """Run backtest with risk management"""
    if signals.empty:
        return {
            'equity': [10000],
            'stats': {
                'final_equity': 10000,
                'total_trades': 0,
                'win_rate': '0.0%',
                'max_drawdown': '0.0%'
            }
        }
    
    equity = 10000
    equity_curve = [equity]
    wins = 0
    peak = equity
    max_drawdown = 0
    
    for _, trade in signals.iterrows():
        try:
            idx = df.index.get_loc(trade['Datetime'])
            entry = trade['Price']
            is_buy = trade['Signal'] == 'Buy'
            
            sl = entry * (1 - sl_pct/100) if is_buy else entry * (1 + sl_pct/100)
            tp = entry * (1 + tp_pct/100) if is_buy else entry * (1 - tp_pct/100)
            
            for i in range(idx, min(idx+100, len(df))):
                current_low = df['Low'].iloc[i]
                current_high = df['High'].iloc[i]
                
                if is_buy:
                    if current_low <= sl:
                        pnl = -sl_pct
                        break
                    elif current_high >= tp:
                        pnl = tp_pct
                        wins += 1
                        break
                else:
                    if current_high >= sl:
                        pnl = -sl_pct
                        break
                    elif current_low <= tp:
                        pnl = tp_pct
                        wins += 1
                        break
            
            equity += (equity * (pnl/100))
            equity_curve.append(equity)
            
            if equity > peak:
                peak = equity
            current_dd = (peak - equity)/peak
            if current_dd > max_drawdown:
                max_drawdown = current_dd
                
        except:
            continue
    
    return {
        'equity': equity_curve,
        'stats': {
            'final_equity': round(equity, 2),
            'total_trades': len(signals),
            'win_rate': f'{wins/max(1,len(signals)):.1%}',
            'max_drawdown': f'{max_drawdown*100:.1f}%'
        }
    }

# ======================
# STREAMLIT APP
# ======================

def main():
    st.set_page_config(
        page_title="NAS100 Trading Assistant",
        layout="wide",
        page_icon="📈"
    )
    
    st.title("📊 NAS100 AI Trading Assistant")
    
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
    else:
        # File upload
        uploaded_file = st.file_uploader("Upload NAS100 Data (CSV)", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['Datetime'], index_col='Datetime')
                st.session_state.df = df
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Process data if available
    if 'df' in st.session_state:
        df = st.session_state.df
        
        # Data validation
        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(df.columns):
            st.error("Missing required price columns (Open, High, Low, Close)")
            st.stop()
        
        # Show raw data
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data Preview")
            st.write(df.head())
        
        # Detect support/resistance
        with st.spinner("Detecting key levels..."):
            support, resistance = detect_support_resistance(df)
            st.success(f"Found {len(support)} support and {len(resistance)} resistance levels")
        
        # Generate signals
        use_volume = st.checkbox("Use Volume Confirmation", value=True)
        signals = generate_signals(df, support, resistance, use_volume)
        
        if not signals.empty:
            # Display signals
            st.subheader("Trade Signals")
            st.dataframe(signals.style.format({'Price': '{:.2f}'}))
            
            # Backtest configuration
            st.subheader("Backtest Parameters")
            col1, col2 = st.columns(2)
            with col1:
                sl_pct = st.slider("Stop Loss %", 0.5, 5.0, 1.5, step=0.1)
            with col2:
                tp_pct = st.slider("Take Profit %", 0.5, 10.0, 3.0, step=0.1)
            
            # Run backtest
            with st.spinner("Running backtest..."):
                result = backtest(df, signals, sl_pct, tp_pct)
            
            # Display results
            st.subheader("Backtest Results")
            col1, col2, col3, col4 = st.columns(4)
            
            final_equity = result['stats']['final_equity']
            equity_change = (final_equity - 10000) / 100
            
            col1.metric("Final Equity", f"${final_equity:,.2f}", f"{equity_change:+.1f}%")
            col2.metric("Win Rate", result['stats']['win_rate'])
            col3.metric("Total Trades", result['stats']['total_trades'])
            col4.metric("Max Drawdown", result['stats']['max_drawdown'])
            
            # Visualizations
            st.subheader("Price Chart")
            st.line_chart(df['Close'])
            
            st.subheader("Equity Curve")
            st.area_chart(pd.DataFrame({'Equity': result['equity']}))
            
            # Export results
            st.subheader("Export Results")
            csv = signals.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(
                f'<a href="data:file/csv;base64,{b64}" download="nas100_signals.csv">'
                '📥 Download Trade Signals</a>',
                unsafe_allow_html=True
            )
            
        else:
            st.warning("No trade signals generated with current parameters")
    
    # Sample data generator
    with st.expander("Need sample data?"):
        sample_data = pd.DataFrame({
            'Datetime': pd.date_range(start='2024-01-01', periods=100, freq='5T'),
            'Open': np.linspace(18000, 18200, 100),
            'High': np.linspace(18005, 18205, 100),
            'Low': np.linspace(17995, 18195, 100),
            'Close': np.linspace(18000, 18200, 100),
            'Volume': np.random.randint(1000, 5000, 100)
        })
        st.download_button(
            "Download Sample Data",
            sample_data.to_csv(index=False),
            "nas100_sample.csv"
        )

if __name__ == "__main__":
    main()
