import pandas as pd
import yfinance as yf

class DataProcess:
    @staticmethod
    def download_data(tickers, start_date, end_date):
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date)
                if df.empty:
                    print(f'No data found for {ticker}.')
                    continue
                df['Ticker'] = ticker
                data[ticker] = df
            except Exception as e:
                print(f'Data fetching error for {ticker}: {e}')
        if not data:
            raise ValueError("Data could not be fetched. Please check ticker symbols and date range.")
        return pd.concat(data.values(), keys=data.keys())

    @staticmethod
    def feature_engineering(data):
        data['SMA_10'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())
        data['SMA_50'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=50).mean())
        data['Return'] = data.groupby('Ticker')['Close'].transform(lambda x: x.pct_change())
        data['Volume'] = data.groupby('Ticker')['Volume'].transform(lambda x: x)
        data['SMA_20'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=20).mean())
        data['Volatility'] = data.groupby('Ticker')['Return'].transform(lambda x: x.rolling(window=10).std())
        data['Momentum'] = data.groupby('Ticker')['Close'].transform(lambda x: x - x.shift(5))
        return data
