import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import requests
from collections import deque

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

def backtest(symbol: str, start_date: datetime, end_date: datetime, 
             initial_balance: float = 10000, overbought: float = 70, 
             oversold: float = 30, use_sma: bool = False, 
             sma_window: int = 50) -> Tuple[pd.DataFrame, List, float, float]:
    data = yf.download(symbol, start=start_date, end=end_date)
    
    data['RSI'] = calculate_rsi(data)
    data['Signal'] = data['RSI'].apply(lambda x: get_signal(x, overbought, oversold))
    
    if use_sma:
        data['SMA'] = calculate_sma(data, sma_window)
        data['Signal'] = np.where((data['Signal'] == 1) & (data['Close'] > data['SMA']), 1,
                                  np.where((data['Signal'] == -1) & (data['Close'] < data['SMA']), -1, 0))
    
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

    fig.update_layout(title='Stock Price with Buy/Sell Signals', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_rsi(data: pd.DataFrame, overbought: float, oversold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
    fig.add_hline(y=overbought, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=oversold, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(title='RSI Indicator', xaxis_title='Date', yaxis_title='RSI')
    return fig

def fetch_news(symbol: str) -> List[Dict]:
    url = f"https://api.stockanalysis.com/api/news/fetch/s/{symbol}/p"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']
    else:
        return []

def display_news(news: List[Dict]):
    for article in news:
        st.subheader(article['title'])
        st.write(f"Source: {article['source']} | {article['time']} ({article['ago']})")
        st.write(article['text'][:300] + "...")  # Display first 300 characters
        st.markdown(f"[Read more]({article['url']})")
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

def main():
    st.set_page_config(page_title="RSI SMA News Stock Screener", layout="wide")
    st.title('RSI SMA News Stock Screener')

    # Load company list
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

    with col2:
        if st.button('Run Backtest and Fetch News'):
            # Fetch news
            with st.spinner('Fetching recent news...'):
                news = fetch_news(symbol)

            # Run backtest
            with st.spinner('Running backtest...'):
                data, trades, final_balance, return_pct = backtest(symbol, start_date, end_date, initial_balance, overbought, oversold, use_sma, sma_window)

            # Display backtest results
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

            # Display news
            st.header(f'Recent News for {symbol}')
            if news:
                display_news(news)
            else:
                st.write("No recent news available.")

            # Correlation analysis between news sentiment and stock performance
            st.header('News Impact Analysis')
            st.write("This section could include an analysis of how news sentiment correlates with stock performance. "
                     "For a more advanced version, we could implement sentiment analysis on the news articles and "
                     "compare it with stock price movements.")

if __name__ == "__main__":
    main()