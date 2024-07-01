import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import requests
from collections import deque
import asyncio
import aiohttp

# Remove appdirs usage
# import appdirs as ad
# ad.user_cache_dir = lambda *args: "/tmp"

# Helper functions
def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_signal(rsi: float, overbought: float, oversold: float) -> int:
    if rsi > overbought:
        return -1  # Sell signal
    elif rsi < oversold:
        return 1  # Buy signal
    return 0  # Hold

def calculate_sma(data: pd.DataFrame, window: int) -> pd.Series:
    return data['Close'].rolling(window=window).mean()

def calculate_macd(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

async def fetch_stock_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    return yf.download(symbol, start=start_date, end=end_date)

def backtest(data: pd.DataFrame, initial_balance: float = 10000, overbought: float = 70, 
             oversold: float = 30, use_sma: bool = False, sma_window: int = 50,
             use_macd: bool = False) -> Tuple[pd.DataFrame, List, float, float]:
    data['RSI'] = calculate_rsi(data)
    data['Signal'] = data['RSI'].apply(lambda x: get_signal(x, overbought, oversold))
    
    if use_sma:
        data['SMA'] = calculate_sma(data, sma_window)
        data['Signal'] = np.where((data['Signal'] == 1) & (data['Close'] > data['SMA']), 1,
                                  np.where((data['Signal'] == -1) & (data['Close'] < data['SMA']), -1, 0))
    
    if use_macd:
        data = calculate_macd(data)
        data['Signal'] = np.where((data['Signal'] == 1) & (data['MACD'] > data['Signal_Line']), 1,
                                  np.where((data['Signal'] == -1) & (data['MACD'] < data['Signal_Line']), -1, data['Signal']))
    
    balance = initial_balance
    shares = 0
    trades = []
    
    for i in range(1, len(data)):
        yesterday, today = data.index[i-1], data.index[i]
        price = data['Close'][today]
        signal = data['Signal'][yesterday]
        
        if signal == 1 and balance > 0:
            shares_to_buy = balance // price
            cost = shares_to_buy * price
            balance -= cost
            shares += shares_to_buy
            trades.append(('BUY', today, shares_to_buy, price))
        elif signal == -1 and shares > 0:
            balance += shares * price
            trades.append(('SELL', today, shares, price))
            shares = 0
    
    final_balance = balance + shares * data['Close'][-1]
    return_pct = (final_balance - initial_balance) / initial_balance * 100
    
    return data, trades, final_balance, return_pct

def plot_backtest_results(data: pd.DataFrame, trades: List) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Stock Price'))
    
    buy_dates = [trade[1] for trade in trades if trade[0] == 'BUY']
    buy_prices = [trade[3] for trade in trades if trade[0] == 'BUY']
    sell_dates = [trade[1] for trade in trades if trade[0] == 'SELL']
    sell_prices = [trade[3] for trade in trades if trade[0] == 'SELL']

    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy', marker=dict(color='green', size=10)))
    fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell', marker=dict(color='red', size=10)))

    if 'SMA' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], name='SMA', line=dict(color='orange', dash='dash')))

    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], name='Signal Line', line=dict(color='red')))

    fig.update_layout(title='Stock Price with Technical Indicators and Buy/Sell Signals', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_rsi(data: pd.DataFrame, overbought: float, oversold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
    fig.add_hline(y=overbought, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=oversold, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(title='RSI Indicator', xaxis_title='Date', yaxis_title='RSI')
    return fig

async def fetch_news(session: aiohttp.ClientSession, symbol: str) -> List[Dict]:
    url = f"https://api.stockanalysis.com/api/news/fetch/s/{symbol}/p"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            return data['data']
        else:
            return []

# Simple sentiment analysis function without using TextBlob
def analyze_sentiment(text: str) -> float:
    # This is a very basic sentiment analysis. For better results, consider using a pre-trained model or API.
    positive_words = ['up', 'rise', 'gain', 'positive', 'growth', 'increase', 'higher']
    negative_words = ['down', 'fall', 'loss', 'negative', 'decline', 'decrease', 'lower']
    
    words = text.lower().split()
    sentiment = sum(word in positive_words for word in words) - sum(word in negative_words for word in words)
    return sentiment / len(words) if words else 0

def display_news(news: List[Dict]):
    for article in news:
        sentiment = analyze_sentiment(article['title'] + " " + article['text'])
        sentiment_color = 'green' if sentiment > 0 else 'red' if sentiment < 0 else 'gray'
        
        st.subheader(article['title'])
        st.write(f"Source: {article['source']} | {article['time']} ({article['ago']})")
        st.write(article['text'][:300] + "...")  # Display first 300 characters
        st.markdown(f"[Read more]({article['url']})")
        st.markdown(f"Sentiment: <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
        st.write("---")

@st.cache_data
def load_company_list():
    # S&P 500 companies
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    sp500_symbols = sp500_table['Symbol'].tolist()
    sp500_names = sp500_table['Security'].tolist()
    
    # Nasdaq 100 companies
    nasdaq_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    nasdaq_table = pd.read_html(nasdaq_url)[4]
    nasdaq_symbols = nasdaq_table['Ticker'].tolist()
    nasdaq_names = nasdaq_table['Company'].tolist()
    
    # Combine and remove duplicates
    all_symbols = sp500_symbols + nasdaq_symbols
    all_names = sp500_names + nasdaq_names
    companies = list(dict.fromkeys(zip(all_symbols, all_names)))
    
    return [f"{symbol} - {name}" for symbol, name in companies]


async def analyze_stock(symbol: str, start_date: datetime, end_date: datetime, initial_balance: float,
                        overbought: float, oversold: float, use_sma: bool, sma_window: int, use_macd: bool) -> Dict:
    try:
        data = await fetch_stock_data(symbol, start_date, end_date)
        data, trades, final_balance, return_pct = backtest(data, initial_balance, overbought, oversold, use_sma, sma_window, use_macd)
        
        return {
            'symbol': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'return_pct': return_pct,
            'num_trades': len(trades),
            'overbought': overbought,
            'oversold': oversold,
            'use_sma': use_sma,
            'sma_window': sma_window if use_sma else 'N/A',
            'use_macd': use_macd
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e)
        }

async def autorun_process(symbols: List[str], start_date: datetime, end_date: datetime, initial_balance: float,
                          overbought: float, oversold: float, use_sma: bool, sma_window: int, use_macd: bool,
                          progress_bar, results_placeholder, stop_event):
    results = []
    for i, symbol in enumerate(symbols):
        if stop_event.is_set():
            break
        
        result = await analyze_stock(symbol, start_date, end_date, initial_balance, overbought, oversold, use_sma, sma_window, use_macd)
        results.append(result)
        
        # Update progress bar and display current result
        progress_bar.progress((i + 1) / len(symbols))
        results_df = pd.DataFrame(results)
        results_placeholder.dataframe(results_df)
        
        # Save results to CSV after each stock analysis
        results_df.to_csv('stock_analysis_results.csv', index=False)
        
        await asyncio.sleep(1)  # Wait for 1 second before analyzing the next stock
    
    return results

def run_single_analysis(symbol: str, start_date: datetime, end_date: datetime, initial_balance: float,
                        overbought: float, oversold: float, use_sma: bool, sma_window: int, use_macd: bool):
    data = yf.download(symbol, start=start_date, end=end_date)
    data, trades, final_balance, return_pct = backtest(data, initial_balance, overbought, oversold, use_sma, sma_window, use_macd)

    st.header('Backtest Results')
    col_results1, col_results2, col_results3 = st.columns(3)
    col_results1.metric("Final Balance", f"${final_balance:.2f}", f"{return_pct:.2f}%")
    col_results2.metric("Number of Trades", len(trades))
    col_results3.metric("Profit/Loss", f"${final_balance - initial_balance:.2f}")

    fig_price = plot_backtest_results(data, trades)
    st.plotly_chart(fig_price, use_container_width=True)

    fig_rsi = plot_rsi(data, overbought, oversold)
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader('Recent Trades')
    st.dataframe(pd.DataFrame(trades[-10:][::-1], columns=['Action', 'Date', 'Shares', 'Price']))

    st.download_button(
        label="Download Full Trade History",
        data=pd.DataFrame(trades, columns=['Action', 'Date', 'Shares', 'Price']).to_csv(index=False).encode('utf-8'),
        file_name=f"{symbol}_trade_history.csv",
        mime="text/csv",
    )

async def autorun_process(symbols: List[str], start_date: datetime, end_date: datetime, initial_balance: float,
                          overbought: float, oversold: float, use_sma: bool, sma_window: int, use_macd: bool,
                          progress_bar, results_placeholder, stop_event):
    results = []
    for i, symbol in enumerate(symbols):
        if stop_event.is_set():
            break
        
        result = await analyze_stock(symbol, start_date, end_date, initial_balance, overbought, oversold, use_sma, sma_window, use_macd)
        results.append(result)
        
        # Update progress bar and display current result
        progress_bar.progress((i + 1) / len(symbols))
        results_df = pd.DataFrame(results)
        results_placeholder.dataframe(results_df)
        
        # Save results to CSV after each stock analysis
        results_df.to_csv('stock_analysis_results.csv', index=False)
        
        await asyncio.sleep(1)  # Wait for 1 second before analyzing the next stock
    
    return results

def main():
    st.set_page_config(page_title="Advanced Stock Analyzer", layout="wide")
    st.title('Advanced Stock Analyzer')

    company_list = load_company_list()

    col1, col2 = st.columns([1, 3])

    with col1:
        st.header('Input Parameters')
        selected_company = st.selectbox('Select a Company', company_list)
        symbol = selected_company.split(' - ')[0]  # Extract symbol from selection
        
        start_date = st.date_input('Start Date', value=datetime.now() - timedelta(days=365))
        end_date = st.date_input('End Date', value=datetime.now())
        initial_balance = st.number_input('Initial Balance ($)', value=10000, min_value=1000, step=1000)
        overbought = st.slider('Overbought Threshold', min_value=50, max_value=90, value=70)
        oversold = st.slider('Oversold Threshold', min_value=10, max_value=50, value=30)
        use_sma = st.checkbox('Use SMA Filter', value=False)
        sma_window = st.number_input('SMA Window', value=50, min_value=5, max_value=200, step=5) if use_sma else 50
        use_macd = st.checkbox('Use MACD Filter', value=False)

        

        st.header('Autorun Options')
        autorun_button = st.button('Start Autorun')

    with col2:
        if st.button('Run Analysis'):
            run_single_analysis(symbol, start_date, end_date, initial_balance, overbought, oversold, use_sma, sma_window, use_macd)

        if autorun_button:
            progress_bar = st.progress(0)
            results_placeholder = st.empty()
            
            # Get all symbols from the company list
            all_symbols = [company.split(' - ')[0] for company in company_list]
            
            stop_event = asyncio.Event()
            stop_button = st.button('Stop Autorun')

            async def run_autorun():
                autorun_task = asyncio.create_task(autorun_process(all_symbols, start_date, end_date, initial_balance,
                                                                   overbought, oversold, use_sma, sma_window, use_macd,
                                                                   progress_bar, results_placeholder, stop_event))
                
                while not autorun_task.done():
                    if stop_button:
                        stop_event.set()
                        st.write("Stopping autorun... Please wait.")
                        break
                    await asyncio.sleep(0.1)
                
                results = await autorun_task
                if stop_event.is_set():
                    st.write("Autorun stopped. Partial results saved to 'stock_analysis_results.csv'")
                else:
                    st.success("Autorun completed. Results saved to 'stock_analysis_results.csv'")
                
                # Display final results, regardless of whether autorun was stopped or completed naturally
                st.dataframe(pd.DataFrame(results))

            st.write("Running autorun... Click 'Stop Autorun' to halt the process.")
            asyncio.run(run_autorun())

if __name__ == "__main__":
    main()