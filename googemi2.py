# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Indian Stock Analysis & Prediction")
warnings.filterwarnings('ignore') # Hide warnings for cleaner output (use cautiously)

# --- Technical Indicator Calculations ---
def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    close_col = 'Close' # Ensure using the correct column name
    if close_col not in data.columns:
        st.warning("RSI calculation requires 'Close' column.")
        return pd.Series(dtype=float) # Return empty series

    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    # Avoid division by zero
    rs = gain / loss
    rs[loss == 0] = np.inf # Handle cases where loss is zero (strong uptrend)
    rsi = 100 - (100 / (1 + rs))
    rsi[(gain == 0) & (loss == 0)] = 50 # Handle cases where both gain and loss are zero (no change)
    rsi[np.isinf(rsi)] = 100 # RSI is 100 if rs is infinite

    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    close_col = 'Close'
    if close_col not in data.columns:
        st.warning("MACD calculation requires 'Close' column.")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    ema_fast = data[close_col].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = data[close_col].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculates Bollinger Bands."""
    close_col = 'Close'
    if close_col not in data.columns:
        st.warning("Bollinger Bands calculation requires 'Close' column.")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    rolling_mean = data[close_col].rolling(window=window, min_periods=1).mean()
    rolling_std = data[close_col].rolling(window=window, min_periods=1).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return upper, rolling_mean, lower # middle band is just the rolling mean

# --- Data Fetching ---
@st.cache_data(ttl=timedelta(minutes=15)) # Cache data for 15 minutes
def get_stock_data(symbol, benchmark=False):
    """Fetches historical stock data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        # Fetch max history for calculations, interval='1d' is default
        data = ticker.history(period="max", auto_adjust=True) # auto_adjust=True handles splits/dividends

        if data.empty:
            if not benchmark: st.error(f"No historical data found for symbol: {symbol}. Check the symbol on Yahoo Finance.")
            return pd.DataFrame(), None # Return empty DataFrame and None for info

        # Standardize column names (lower case, then capitalize first letter)
        data.columns = data.columns.str.lower()
        data.rename(columns=str.capitalize, inplace=True) # Ensure consistent naming like 'Close'

        # Check for essential 'Close' column AFTER standardization
        if 'Close' not in data.columns:
             if not benchmark: st.error(f"Essential 'Close' price data missing for {symbol} after standardization.")
             return pd.DataFrame(), None

        # Ensure index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
             data.index = pd.to_datetime(data.index)

        # Get Ticker Info (only if not fetching benchmark)
        ticker_info = {}
        if not benchmark:
            try:
                ticker_info = ticker.info
                # Sometimes info is empty or lacks keys, handle this gracefully later using .get()
            except Exception as info_e:
                 st.warning(f"Could not fetch detailed info for {symbol}: {info_e}")

        # Select common columns if they exist, prioritizing standardized names
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in data.columns]

        # Make sure index is timezone naive for easier comparisons later
        if data.index.tz is not None:
             data.index = data.index.tz_localize(None)

        return data[available_cols], ticker_info # Return data and info dict

    except Exception as e:
        if not benchmark: st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame(), None

@st.cache_data(ttl=timedelta(hours=1))
def get_ticker_details(symbol):
    """Fetches detailed ticker info, news, dividends, holders separately for better caching/error handling."""
    news, dividends, major_holders = [], pd.Series(dtype=float), pd.DataFrame() # Default empty values
    try:
        ticker = yf.Ticker(symbol)
        try: news = ticker.news
        except Exception as e: st.warning(f"Could not fetch news for {symbol}: {e}")
        try: dividends = ticker.dividends
        except Exception as e: st.warning(f"Could not fetch dividends for {symbol}: {e}")
        try: major_holders = ticker.major_holders
        except Exception as e: st.warning(f"Could not fetch major holders for {symbol}: {e}")

        # If dividends is a Series, ensure its index is timezone naive
        if isinstance(dividends, pd.Series) and dividends.index.tz is not None:
            dividends.index = dividends.index.tz_localize(None)

        return news, dividends, major_holders
    except Exception as e:
        st.warning(f"Could not initialize Ticker object for details ({symbol}): {e}")
        return news, dividends, major_holders # Return defaults

# --- Investment Calculator ---
def calculate_investment_return(principal, annual_rate, years):
    """Calculates future value of an investment using compound interest."""
    if not all(isinstance(i, (int, float)) for i in [principal, annual_rate, years]):
        return 0, 0
    if principal <= 0 or annual_rate <= 0 or years <= 0:
        return 0, 0
    rate = annual_rate / 100.0
    future_value = principal * ((1 + rate) ** years)
    total_profit = future_value - principal
    return future_value, total_profit

# --- Risk Metrics Calculation ---
def calculate_volatility_beta(stock_data, benchmark_data, window=252): # 252 trading days in a year
    """Calculates annualized volatility and beta."""
    # Basic validation
    if stock_data is None or benchmark_data is None or stock_data.empty or benchmark_data.empty:
        # st.warning("Cannot calculate Volatility/Beta: Missing stock or benchmark data.")
        return None, None
    if 'Close' not in stock_data.columns or 'Close' not in benchmark_data.columns:
        # st.warning("Cannot calculate Volatility/Beta: 'Close' column missing.")
        return None, None

    # Align data by date index (important!)
    df_merged = pd.merge(stock_data[['Close']], benchmark_data[['Close']], left_index=True, right_index=True, how='inner', suffixes=('_Stock', '_Bench'))
    df_merged.rename(columns={'Close_Stock': 'Stock', 'Close_Bench': 'Benchmark'}, inplace=True)

    if len(df_merged) < 2 : # Need at least two points to calculate returns
         # st.warning(f"Cannot calculate Volatility/Beta: Insufficient overlapping data ({len(df_merged)} points).")
         return None, None # Not enough data for reliable calculation

    returns = df_merged.pct_change().dropna()

    if returns.empty:
        # st.warning("Cannot calculate Volatility/Beta: No valid returns after alignment.")
        return None, None

    # Use a minimum calculation period (e.g., 30 days) or the window, whichever is smaller if data is limited
    calc_window = min(window, len(returns))
    if calc_window < 30: # Need a reasonable minimum number of returns
        # st.warning(f"Cannot calculate Volatility/Beta: Not enough returns ({calc_window}) for stable calculation.")
        return None, None

    # Volatility (Annualized Standard Deviation of Stock Returns)
    # Calculate using the available returns within the calc_window
    stock_returns_for_calc = returns['Stock'].iloc[-calc_window:]
    latest_vol = stock_returns_for_calc.std() * np.sqrt(window) # Annualize using standard window

    # Beta (Covariance(Stock, Benchmark) / Variance(Benchmark))
    returns_for_calc = returns.iloc[-calc_window:]
    cov = returns_for_calc.cov().iloc[0, 1] # Covariance between Stock and Benchmark
    var = returns_for_calc['Benchmark'].var()

    latest_beta = cov / var if var != 0 else None

    # Check for NaN results which can happen if std dev or variance is zero
    if pd.isna(latest_vol) or pd.isna(latest_beta):
        # st.warning("Volatility/Beta calculation resulted in NaN, possibly due to zero variance.")
        return None, None

    return latest_vol, latest_beta


# --- App Layout ---

# Title
st.title("📈 Indian Stock Analysis & Prediction")
st.markdown("*(Data provided by Yahoo Finance)*")

# Sidebar for user inputs
st.sidebar.header("⚙️ User Input")
stock_symbol = st.sidebar.text_input(
    "Enter Indian Stock Symbol (e.g., RELIANCE.NS, TCS.NS)",
    "RELIANCE.NS"
).upper() # Convert to uppercase

# Date range selector
st.sidebar.subheader("🗓️ Date Range for Analysis")
end_date_default = datetime.today().date() # Use date() object
start_date_default = end_date_default - timedelta(days=365)

col1_date, col2_date = st.sidebar.columns(2)
with col1_date:
    start_date = st.date_input(
        "Start Date",
        value=start_date_default,
        max_value=end_date_default - timedelta(days=1) # Cannot be today
    )
with col2_date:
    end_date = st.date_input(
        "End Date",
        value=end_date_default,
        min_value=start_date + timedelta(days=1), # Must be after start
        max_value=end_date_default # Can be today
    )

# Convert date objects to strings for slicing DataFrames
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')


# Technical indicators
st.sidebar.subheader("📊 Technical Indicators")
show_ma50 = st.sidebar.checkbox("50-Day MA", True)
show_ma200 = st.sidebar.checkbox("200-Day MA", True)
show_rsi = st.sidebar.checkbox("RSI (14-day)", True)
show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands (20, 2)", True) # Default BB to True

# Prediction days
st.sidebar.subheader("🔮 Prediction")
predict_days = st.sidebar.slider("Days to Predict Ahead", 1, 90, 15) # Default to 15 days

# --- Main Panel ---

# Fetch Stock Data and Benchmark Data
nifty_symbol = "^NSEI" # Nifty 50 symbol

# --- Fetch Full Data First ---
stock_data_full, stock_info = get_stock_data(stock_symbol)
nifty_data_full, _ = get_stock_data(nifty_symbol, benchmark=True) # Fetch Nifty data

# Fetch other details (news, dividends etc.) - Do this after confirming stock_symbol is valid
news_data, dividend_data, holder_data = [], pd.Series(dtype=float), pd.DataFrame() # Initialize
if stock_data_full is not None and not stock_data_full.empty:
    news_data, dividend_data, holder_data = get_ticker_details(stock_symbol)
else:
    # Error message was already shown in get_stock_data
    st.stop() # Stop execution if primary stock data failed

# --- Ensure stock data is usable ---
if stock_data_full is None or stock_data_full.empty or 'Close' not in stock_data_full.columns:
    st.error(f"Could not retrieve valid data (including 'Close' price) for symbol: {stock_symbol}. Please check the symbol and try again.")
    st.stop() # Stop execution if data is invalid

# --- Calculate Indicators on Full Dataset BEFORE filtering ---
# Make copies to avoid SettingWithCopyWarning if needed later
stock_data_full = stock_data_full.copy()
if 'Close' in stock_data_full.columns:
    if show_ma50: stock_data_full['MA_50'] = stock_data_full['Close'].rolling(50, min_periods=1).mean()
    if show_ma200: stock_data_full['MA_200'] = stock_data_full['Close'].rolling(200, min_periods=1).mean()
    if show_rsi: stock_data_full['RSI'] = calculate_rsi(stock_data_full)
    if show_macd:
        macd_line, signal_line, hist = calculate_macd(stock_data_full)
        stock_data_full['MACD'] = macd_line
        stock_data_full['MACD_Signal'] = signal_line
        stock_data_full['MACD_Hist'] = hist
    # *** ADDED BOLLINGER BANDS CALCULATION HERE ***
    if show_bollinger:
        upper, middle, lower = calculate_bollinger_bands(stock_data_full)
        stock_data_full['Upper_Band'] = upper
        stock_data_full['Middle_Band'] = middle # SMA(20)
        stock_data_full['Lower_Band'] = lower

# --- Filter data based on selected date range ---
# Use string dates for loc slicing which is inclusive
try:
    stock_data = stock_data_full.loc[start_date_str:end_date_str].copy()
except KeyError:
     st.warning(f"Date range ({start_date_str} to {end_date_str}) might be outside available data. Adjusting to available range.")
     # Attempt to filter using the index directly if loc fails
     stock_data = stock_data_full[(stock_data_full.index >= pd.Timestamp(start_date_str)) & (stock_data_full.index <= pd.Timestamp(end_date_str))].copy()


# --- Ensure filtered data is not empty ---
if stock_data.empty:
    st.warning(f"No data available for {stock_symbol} in the selected date range ({start_date_str} to {end_date_str}). Try adjusting the dates or checking the symbol.")
    st.stop()

# --- Ensure sufficient data for analysis after filtering ---
if len(stock_data) < 2:
    st.warning(f"Insufficient data ({len(stock_data)} points) in the selected date range for meaningful analysis. Please select a wider date range.")
    st.stop()

# --- Calculate Risk Metrics ---
# Use data up to the selected end_date for calculation
stock_data_for_risk = stock_data_full.loc[:end_date_str] # Use full history up to end date
nifty_data_for_risk = nifty_data_full.loc[:end_date_str] if nifty_data_full is not None and not nifty_data_full.empty else None
annual_volatility, beta = calculate_volatility_beta(stock_data_for_risk, nifty_data_for_risk)


# --- Dashboard Header ---
# Use .get() extensively for stock_info as it might be partially filled or empty
company_name = stock_info.get('longName', stock_symbol) if stock_info else stock_symbol
st.header(f"{company_name} ({stock_symbol})")
if stock_info:
    st.caption(f"Sector: **{stock_info.get('sector', 'N/A')}** | Industry: **{stock_info.get('industry', 'N/A')}** | Currency: **{stock_info.get('currency', 'N/A')}**")
else:
     st.caption("Detailed company info not available.")


# --- Tabs for Different Views ---
tab_summary, tab_chart, tab_technicals, tab_fundamentals, tab_news, tab_prediction, tab_calculator = st.tabs([
    "📊 Summary", "📈 Chart", "📉 Technicals", "🏢 Fundamentals", "📰 News", "🔮 Prediction", "💰 Calculator"
])

# --- Summary Tab ---
with tab_summary:
    st.subheader("Key Metrics & Performance")
    latest = stock_data.iloc[-1]
    prev_close = stock_data.iloc[-2]['Close'] if len(stock_data) > 1 else latest['Close'] # Safe previous close
    delta = latest['Close'] - prev_close
    delta_pct = (delta / prev_close) * 100 if prev_close != 0 else 0

    # Get currency symbol or default
    currency = stock_info.get('currency', '₹') if stock_info else '₹'

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Last Close Price", f"{currency}{latest['Close']:,.2f}", f"{delta:,.2f} ({delta_pct:.2f}%)")
        if 'Volume' in latest and pd.notna(latest['Volume']): st.metric("Volume", f"{latest['Volume']:,.0f}")
        else: st.metric("Volume", "N/A")
        if annual_volatility is not None: st.metric("Annualized Volatility", f"{annual_volatility:.2%}")
        else: st.metric("Annualized Volatility", "N/A")

    with col_s2:
        if 'Open' in latest and pd.notna(latest['Open']): st.metric("Open", f"{currency}{latest['Open']:,.2f}")
        else: st.metric("Open", "N/A")
        # Prefer yfinance info for day high/low if available
        day_high_yf = stock_info.get('dayHigh') if stock_info else None
        day_high_hist = latest.get('High') if 'High' in latest else None
        day_high = day_high_yf if day_high_yf is not None else day_high_hist
        if day_high is not None and pd.notna(day_high): st.metric("Day High", f"{currency}{day_high:,.2f}")
        else: st.metric("Day High", "N/A")

        if beta is not None: st.metric("Beta (vs Nifty 50)", f"{beta:.2f}")
        else: st.metric("Beta (vs Nifty 50)", "N/A")

    with col_s3:
         market_cap = stock_info.get('marketCap') if stock_info else None
         if market_cap is not None: st.metric("Market Cap", f"{currency}{market_cap:,.0f}")
         else: st.metric("Market Cap", "N/A")

         day_low_yf = stock_info.get('dayLow') if stock_info else None
         day_low_hist = latest.get('Low') if 'Low' in latest else None
         day_low = day_low_yf if day_low_yf is not None else day_low_hist
         if day_low is not None and pd.notna(day_low): st.metric("Day Low", f"{currency}{day_low:,.2f}")
         else: st.metric("Day Low", "N/A")

         pe_ratio = stock_info.get('trailingPE') if stock_info else None
         if pe_ratio is not None and pd.notna(pe_ratio): st.metric("P/E Ratio (TTM)", f"{pe_ratio:.2f}")
         else: st.metric("P/E Ratio (TTM)", "N/A")

    st.divider()
    st.subheader(f"Performance in Selected Range ({start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')})")
    # Use .get() for safety in case columns were missing
    period_high = stock_data.get('High', stock_data['Close']).max()
    period_low = stock_data.get('Low', stock_data['Close']).min()
    period_start_price = stock_data.get('Open', stock_data['Close']).iloc[0]
    period_end_price = stock_data['Close'].iloc[-1]
    period_return = ((period_end_price - period_start_price) / period_start_price) * 100 if period_start_price != 0 else 0

    col_p1, col_p2, col_p3 = st.columns(3)
    if pd.notna(period_high):
        with col_p1:
            st.metric("Period High", f"{currency}{period_high:,.2f}")
    else:
        with col_p1:
            st.metric("Period High", "N/A")
    if pd.notna(period_low):
        with col_p2:
            st.metric("Period Low", f"{currency}{period_low:,.2f}")
    else:
        with col_p2:
            st.metric("Period Low", "N/A")
    if pd.notna(period_return):
        with col_p3:
            st.metric("Period Return", f"{period_return:.2f}%")
    else:
        with col_p3:
            st.metric("Period Return", "N/A")

    st.divider()
    st.subheader("52-Week Performance")
    high_52wk, low_52wk = None, None # Initialize
    try:
        # Use full history up to the end date to find the latest 52 weeks (approx 365 days)
        data_1y = stock_data_full.loc[:end_date_str].last('365D')
        if data_1y is not None and not data_1y.empty and len(data_1y)>1:
            high_52wk = data_1y.get('High', data_1y['Close']).max()
            low_52wk = data_1y.get('Low', data_1y['Close']).min()

            col_52w_1, col_52w_2 = st.columns(2)
            if pd.notna(high_52wk):
                with col_52w_1:
                    st.metric("52-Week High", f"{currency}{high_52wk:,.2f}")
            else:
                with col_52w_1:
                    st.metric("52-Week High", "N/A")
            if pd.notna(low_52wk):
                with col_52w_2:
                    st.metric("52-Week Low", f"{currency}{low_52wk:,.2f}")
            else:
                with col_52w_2:
                    st.metric("52-Week Low", "N/A")
        else:
            st.info("Not enough historical data (less than 365 days or 2 points) available for 52-week range calculation.")
    except Exception as e:
        st.error(f"Error calculating 52-week performance: {e}")

# --- Chart Tab ---
with tab_chart:
    st.subheader("Interactive Price Chart")
    chart_type = st.radio("Chart Type:", ["Line", "Candlestick"], index=1, horizontal=True, key="price_chart_type_main")

    # Create figure with secondary y-axis for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      vertical_spacing=0.03, row_heights=[0.75, 0.25],
                      specs=[[{"secondary_y": False}], [{"secondary_y": False}]]) # Simpler spec

    # Add Price Trace (Candlestick or Line)
    # Check necessary columns exist in the filtered stock_data
    ohlc_present = all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close'])

    if chart_type == "Candlestick":
        if ohlc_present:
            fig.add_trace(go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'],
                                         name='Candlestick'), row=1, col=1)
        else:
            st.warning("Candlestick requires Open, High, Low, Close data. Falling back to Line chart.")
            if 'Close' in stock_data:
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close', line=dict(color='#1f77b4')), row=1, col=1)
            else:
                 st.error("Cannot plot price: 'Close' data is missing.")

    else: # Line Chart requested
         if 'Close' in stock_data:
             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close', line=dict(color='#1f77b4')), row=1, col=1)
         else:
             st.error("Cannot plot price: 'Close' data is missing.")


    # Add Moving Averages (Check existence in filtered stock_data)
    if show_ma50 and 'MA_50' in stock_data: fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_50'], name='MA 50', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
    if show_ma200 and 'MA_200' in stock_data: fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_200'], name='MA 200', line=dict(color='#d62728', width=1.5)), row=1, col=1)

    # Add Bollinger Bands (Check existence in filtered stock_data)
    if show_bollinger and all(c in stock_data for c in ['Upper_Band', 'Lower_Band', 'Middle_Band']):
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_Band'], name='Upper BB', line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)
        # Plot middle band (SMA 20) as well, useful reference
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Middle_Band'], name='Middle BB (SMA 20)', line=dict(color='#aec7e8', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_Band'], name='Lower BB', line=dict(color='#2ca02c', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(44, 160, 44, 0.1)'), row=1, col=1) # Fill to upper band trace

    # Add 52 Week High/Low Lines (Support/Resistance) - Use values calculated earlier
    if high_52wk is not None and pd.notna(high_52wk):
         fig.add_hline(y=high_52wk, line_dash="dot", line_color="firebrick", line_width=1.5, annotation_text=f"52W High ({currency}{high_52wk:,.2f})", annotation_position="bottom right", row=1, col=1)
    if low_52wk is not None and pd.notna(low_52wk):
         fig.add_hline(y=low_52wk, line_dash="dot", line_color="green", line_width=1.5, annotation_text=f"52W Low ({currency}{low_52wk:,.2f})", annotation_position="top right", row=1, col=1)

    # Add Volume bars
    if 'Volume' in stock_data and 'Close' in stock_data and 'Open' in stock_data:
        # Determine color based on price change
        colors = ['#2ca02c' if row['Close'] >= row['Open'] else '#d62728' for index, row in stock_data.iterrows()]
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    elif 'Volume' in stock_data:
        # Plot volume without color coding if Open is missing
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', marker_color='grey', opacity=0.7), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)


    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False, # Hide range slider on bottom plot
        yaxis_title=f"Price ({currency})", # Add currency to axis
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", # Better hover info
        margin=dict(l=50, r=50, t=50, b=50) # Adjust margins
    )
    # Add spike lines for better crosshair tracking
    fig.update_xaxes(showspikes=True, spikecolor="grey", spikethickness=1, row=1, col=1)
    fig.update_xaxes(showspikes=True, spikecolor="grey", spikethickness=1, row=2, col=1)
    fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=1, row=1, col=1)
    fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=1, row=2, col=1)

    # Ensure axis titles are set correctly
    fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    if 'Volume' in stock_data:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# --- Technical Indicators Tab ---
with tab_technicals:
    st.subheader("Technical Indicators")

    if not (show_rsi or show_macd):
        st.info("No technical indicators selected in the sidebar.")
    else:
        tech_tabs = st.tabs([tab for tab, show in [("RSI", show_rsi), ("MACD", show_macd)] if show])
        tab_idx = 0

        if show_rsi:
            with tech_tabs[tab_idx]:
                if 'RSI' in stock_data:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], name='RSI', line=dict(color='#9467bd')))
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top right")
                    fig_rsi.update_layout(title="Relative Strength Index (RSI)", yaxis_title="RSI Value", height=350, hovermode="x unified")
                    fig_rsi.update_xaxes(showspikes=True)
                    fig_rsi.update_yaxes(range=[0, 100], showspikes=True)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                else: st.warning("RSI data not available or calculated for the selected period.")
            tab_idx += 1

        if show_macd:
            with tech_tabs[tab_idx]:
                if all(c in stock_data for c in ['MACD', 'MACD_Signal', 'MACD_Hist']):
                    fig_macd = make_subplots(rows=1, cols=1)
                    # Plot Lines First
                    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], name='MACD', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
                    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD_Signal'], name='Signal', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
                    # Plot Histogram
                    colors_macd = ['#2ca02c' if val >= 0 else '#d62728' for val in stock_data['MACD_Hist']]
                    fig_macd.add_trace(go.Bar(x=stock_data.index, y=stock_data['MACD_Hist'], name='Histogram', marker_color=colors_macd, opacity=0.7), row=1, col=1)
                    fig_macd.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
                    fig_macd.update_layout(title="MACD (Moving Average Convergence Divergence)", yaxis_title="Value", height=350, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    fig_macd.update_xaxes(showspikes=True)
                    fig_macd.update_yaxes(showspikes=True)
                    st.plotly_chart(fig_macd, use_container_width=True)
                else: st.warning("MACD data not available or calculated for the selected period.")
            tab_idx += 1


# --- Fundamentals Tab ---
with tab_fundamentals:
    st.subheader("Company Overview & Fundamentals")

    if stock_info: # Check if stock_info dictionary has data
        # Company Profile Expander
        with st.expander("Company Profile", expanded=True):
            col_f1, col_f2 = st.columns([3, 1]) # Give more space to description
            with col_f1:
                st.markdown(f"**{stock_info.get('longName', 'N/A')}**")
                st.caption(f"Sector: `{stock_info.get('sector', 'N/A')}` | Industry: `{stock_info.get('industry', 'N/A')}` | Employees: `{stock_info.get('fullTimeEmployees', 'N/A')}`")
                st.markdown("**Business Summary:**")
                # Use markdown for potentially long text with better wrapping
                st.markdown(f"<p style='text-align: justify;'>{stock_info.get('longBusinessSummary', 'N/A')}</p>", unsafe_allow_html=True)

            with col_f2:
                website = stock_info.get('website', 'N/A')
                if website != 'N/A' and website is not None:
                     # Add protocol if missing for link_button
                     if not website.startswith(('http://', 'https://')):
                         website = 'https://' + website
                     st.link_button("Visit Website 🔗", website)

                st.markdown("**Address:**")
                address_parts = [stock_info.get(k) for k in ['address1', 'address2', 'city', 'state', 'zip'] if stock_info.get(k)]
                country = stock_info.get('country')
                if country: address_parts.append(country)
                st.caption("\n".join(address_parts) if address_parts else "N/A")

                phone = stock_info.get('phone', 'N/A')
                st.caption(f"Phone: {phone}")

        st.divider()
        # Key Financial Ratios Expander
        with st.expander("Key Financial Ratios", expanded=False): # Start collapsed
             col_r1, col_r2, col_r3, col_r4 = st.columns(4)
             currency = stock_info.get('currency', '') # Use currency symbol where appropriate

             def display_ratio(column, label, key, format_str="{:.2f}", currency_prefix=False, na_val='N/A'):
                  value = stock_info.get(key)
                  if value is not None and pd.notna(value): # Check for None and NaN
                      formatted_value = format_str.format(value)
                      if currency_prefix: formatted_value = f"{currency}{formatted_value}"
                      column.metric(label, formatted_value)
                  else:
                      column.metric(label, na_val)

             display_ratio(col_r1, "Market Cap", 'marketCap', format_str="{:,.0f}", currency_prefix=True)
             display_ratio(col_r1, "Enterprise Value", 'enterpriseValue', format_str="{:,.0f}", currency_prefix=True)
             display_ratio(col_r1, "Shares Outstanding", 'sharesOutstanding', format_str="{:,.0f}")

             display_ratio(col_r2, "P/E Ratio (Trailing)", 'trailingPE')
             display_ratio(col_r2, "P/E Ratio (Forward)", 'forwardPE')
             display_ratio(col_r2, "PEG Ratio", 'pegRatio')

             display_ratio(col_r3, "Price-to-Sales (TTM)", 'priceToSalesTrailing12Months')
             display_ratio(col_r3, "Price-to-Book", 'priceToBook')
             display_ratio(col_r3, "Enterprise/Revenue", 'enterpriseToRevenue')


             display_ratio(col_r4, "EPS (TTM)", 'trailingEps', currency_prefix=True)
             display_ratio(col_r4, "EPS (Forward)", 'forwardEps', currency_prefix=True)
             display_ratio(col_r4, "Beta", 'beta') # Beta from yfinance info

        st.divider()
        # Dividend Information Expander
        with st.expander("Dividend Information", expanded=False):
             col_d1, col_d2, col_d3, col_d4 = st.columns(4)
             currency = stock_info.get('currency', '₹')
             display_ratio(col_d1, "Dividend Rate", 'dividendRate', currency_prefix=True, na_val='N/A')
             display_ratio(col_d2, "Dividend Yield", 'dividendYield', format_str="{:.2%}", na_val='N/A') # Format as percentage
             display_ratio(col_d3, "Payout Ratio", 'payoutRatio', format_str="{:.2%}", na_val='N/A')
             ex_div_ts = stock_info.get('exDividendDate')
             if ex_div_ts is not None and pd.notna(ex_div_ts):
                 ex_div_date = datetime.fromtimestamp(ex_div_ts).strftime('%Y-%m-%d')
                 col_d4.metric("Ex-Dividend Date", ex_div_date)
             else:
                 col_d4.metric("Ex-Dividend Date", "N/A")


             if dividend_data is not None and not dividend_data.empty:
                 st.write("**Recent Dividend Payments:**")
                 # Display only recent dividends (e.g., last 5 years)
                 recent_dividends = dividend_data[dividend_data.index > (datetime.now() - timedelta(days=5*365))]
                 if not recent_dividends.empty:
                     # If recent_dividends is a Series, convert it to a DataFrame
                     if isinstance(recent_dividends, pd.Series):
                         recent_dividends = recent_dividends.to_frame(name='Dividends')

                     st.dataframe(recent_dividends.tail().sort_index(ascending=False).style.format("₹{:.2f}"), use_container_width=True)
                 else:
                     st.info("No dividend payments found in the last 5 years.")
             else:
                 st.info("No dividend data found for this stock.")

        st.divider()
        # Shareholding Pattern Expander
        with st.expander("Shareholding Pattern", expanded=False):
             if holder_data is not None and not holder_data.empty:
                 st.write("**Major Holders:**")
                 # Rename columns for clarity if they exist
                 holder_data_display = holder_data.copy()
                 rename_map = {
                     'Insider Pct Held': '% Held by Insiders',
                     'Institution Pct Held': '% Held by Institutions',
                     'Date Reported': 'Reported Date',
                     '% Out': '% of Outstanding'
                 }
                 holder_data_display.columns = [rename_map.get(col, col) for col in holder_data_display.columns]

                 # Select and format columns for better display
                 display_cols = ['Holder', 'Shares', '% of Outstanding', 'Reported Date', 'Value']
                 available_display_cols = [col for col in display_cols if col in holder_data_display.columns]

                 if available_display_cols:
                     st.dataframe(
                         holder_data_display[available_display_cols].style.format({
                             'Shares': '{:,.0f}',
                             'Value': f"{currency}{{:,.0f}}",
                             '% of Outstanding': '{:.2%}'
                         }, na_rep='N/A'),
                         use_container_width=True
                     )
                 else:
                     st.dataframe(holder_data_display) # Show raw if expected cols are missing
             else:
                 st.info("Major holders data not available.")

    else:
        st.warning("Detailed fundamental data could not be retrieved for this stock (ticker.info might be empty or failed).")


# --- News Tab ---
with tab_news:
    st.subheader(f"Recent News for {stock_symbol}")
    try:
        if news_data: # Check if list is not empty
            for i, item in enumerate(news_data):
                title = item.get('title', 'No Title')
                link = item.get('link', '#')
                publisher = item.get('publisher', 'N/A')
                publish_time_ts = item.get('providerPublishTime')
                if publish_time_ts:
                    publish_time = datetime.fromtimestamp(publish_time_ts).strftime('%Y-%m-%d %H:%M')
                else:
                    publish_time = 'N/A'

                st.markdown(f"**[{title}]({link})**")
                st.caption(f"Publisher: {publisher} | Published: {publish_time}")
                # Optional: Add summary if needed and available
                # summary = item.get('summary', '')
                # if summary: st.write(summary)
                if i < len(news_data) - 1: st.divider() # Add divider between news items
        else:
            st.info(f"No recent news found for {stock_symbol} via Yahoo Finance.")
    except Exception as e:
        st.error(f"An error occurred while displaying news: {e}")


# --- Prediction Tab ---
with tab_prediction:
    st.subheader(f"🔮 Price Prediction (Next {predict_days} Trading Days)")
    st.info("ℹ️ **Note:** This is a *very basic* linear regression prediction based on the recent historical trend (using day number as the feature). It does **not** account for market news, fundamentals, technical indicators, volatility, or future events. **Use with extreme caution and only for illustrative purposes.**")

    # Use full historical data up to the selected end date for training basis
    prediction_base_data = stock_data_full[['Close']].loc[:end_date_str].copy()
    prediction_base_data = prediction_base_data.sort_index(ascending=True).dropna()

    if len(prediction_base_data) < 50: # Check after cleaning
        st.warning(f"Insufficient historical data (< 50 valid days ending {end_date_str}) for a meaningful linear trend projection.")
        # st.stop() # Don't stop the whole app, just skip prediction
    else:
        try:
            # Prepare data
            prediction_base_data['Days'] = np.arange(len(prediction_base_data))

            # Train model (using last N points or available, e.g. 100)
            train_window = min(100, len(prediction_base_data))
            train_data = prediction_base_data.iloc[-train_window:]
            X_train = train_data[['Days']] # Keep as DataFrame
            y_train = train_data['Close'].values

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict future days
            last_day_index_num = prediction_base_data['Days'].iloc[-1]
            future_days_indices_num = np.arange(last_day_index_num + 1, last_day_index_num + 1 + predict_days)
            future_days_df = pd.DataFrame({'Days': future_days_indices_num}) # Create DataFrame for prediction
            future_prices = model.predict(future_days_df)

            # Create future dates (business days)
            last_date = prediction_base_data.index[-1]
            # Generate business days, ensuring they start *after* the last known date
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=predict_days)

            # Check if prediction length matches date length
            if len(future_dates) != len(future_prices):
                 st.warning(f"Prediction length mismatch ({len(future_prices)}) vs Business Dates ({len(future_dates)}). Truncating.")
                 min_len = min(len(future_dates), len(future_prices))
                 future_dates = future_dates[:min_len]
                 future_prices = future_prices[:min_len]


            predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_prices}).set_index('Date')

            # --- Prediction Plot ---
            st.markdown(f"**Predicted Trend based on Linear Regression:**")
            fig_pred = go.Figure()

            # Plot recent historical data (e.g., last 60 points from filtered data)
            hist_window = min(60, len(stock_data))
            historical_plot_data = stock_data.iloc[-hist_window:]
            fig_pred.add_trace(go.Scatter(x=historical_plot_data.index, y=historical_plot_data['Close'], mode='lines', name='Actual (Recent)', line=dict(color='#1f77b4')))

            # Plot prediction
            fig_pred.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['Predicted Close'], mode='lines', name='Predicted Trend', line=dict(color='#ff7f0e', dash='dot')))

            fig_pred.update_layout(
                title=f"{stock_symbol} - Price vs. {predict_days}-Day Linear Trend Projection",
                xaxis_title="Date",
                yaxis_title=f"Price ({currency})",
                hovermode="x unified",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_pred.update_xaxes(showspikes=True)
            fig_pred.update_yaxes(showspikes=True)
            st.plotly_chart(fig_pred, use_container_width=True)

            # Display predicted values table
            st.markdown("**Projected Closing Prices (Linear Trend):**")
            st.dataframe(predictions_df.style.format({"Predicted Close": f"{currency}{{:,.2f}}" }))

        except Exception as e:
            st.error(f"An error occurred during the prediction calculation or plotting: {e}")
            st.warning("Prediction could not be generated.")


# --- Investment Calculator Tab ---
with tab_calculator:
    st.subheader("💰 Investment Return Calculator (Hypothetical)")
    st.write("Estimate potential returns based on an *assumed constant* average annual growth rate (compounded annually).")

    col_calc1, col_calc2, col_calc3 = st.columns(3)
    with col_calc1: initial_investment = st.number_input("Initial Investment (₹)", min_value=1.0, value=10000.0, step=1000.0, format="%.2f", key="inv_amount")
    with col_calc2: annual_growth_rate = st.number_input("Assumed Annual Growth Rate (%)", min_value=0.0, value=12.0, step=0.5, format="%.1f", key="inv_rate")
    with col_calc3: investment_years = st.number_input("Investment Horizon (Years)", min_value=1, value=5, step=1, key="inv_years") # Changed to number input


    if initial_investment > 0 and annual_growth_rate >= 0 and investment_years > 0:
        future_val, profit = calculate_investment_return(initial_investment, annual_growth_rate, investment_years)
        st.markdown(f"**After {investment_years} years:**")
        col_res1, col_res2 = st.columns(2)
        with col_res1: st.metric(label="Estimated Future Value", value=f"₹{future_val:,.2f}")
        with col_res2: st.metric(label="Estimated Total Profit", value=f"₹{profit:,.2f}")
        st.warning("⚠️ **Disclaimer:** This calculator provides a simple projection based *only* on the assumed constant compound growth rate. It does **NOT** reflect real-world stock market volatility, fees, taxes, inflation, or guarantee any returns. Actual investment outcomes can vary significantly.")
    else:
        st.info("Enter a valid investment amount (>0), growth rate (>=0%), and horizon (>0) to calculate.")

# --- Final Disclaimer ---
st.divider()
st.warning("🛑 **Overall Disclaimer:** This application is for informational and educational purposes only. Analysis, indicators, and predictions are based on historical data and standard models, **not financial advice**. Stock market investments involve significant risk, including the potential loss of principal. Always conduct thorough independent research and/or consult with a qualified financial professional before making any investment decisions.")