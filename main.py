from datetime import datetime
from Finance.DataProcess import DataProcess
from Finance.DataModel import DataModel


def main():
    tickers = ['your-stock-shortname.IS']
    start_date = '2022-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        data = DataProcess.download_data(tickers, start_date, end_date)
    except ValueError as e:
        print(e)
        return

    data = DataProcess.feature_engineering(data)
    X, y = DataModel.prepare_data(data)

    best_model, scaler = DataModel.train_model(X, y)

    latest_data = data.loc[data['Ticker'] == tickers[0]].iloc[-1]
    future_price = DataModel.predict_future_price(best_model, scaler, latest_data, days_ahead=0)

    print(f'Future Price ({end_date}): {future_price}')

    user_date = input("Enter the date you want to make a prediction for (YYYY-MM-DD): ")

    try:
        # Check date format
        datetime.strptime(user_date, '%Y-%m-%d')

        user_data = DataProcess.download_data(tickers, user_date, user_date)
        user_data = DataProcess.feature_engineering(user_data)

        if len(user_data) > 0:
            latest_user_data = user_data.iloc[-1]
            user_future_price = DataModel.predict_future_price(best_model, scaler, latest_user_data, days_ahead=0)
            print(f'Predicted price for date {user_date}: {user_future_price}')
        else:
            print(f"No data found for date {user_date}.")
    except ValueError as e:
        print(f"Date format error: {e}. Please enter a date in 'YYYY-MM-DD' format.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

    # Short-term and long-term predictions
    short_term_price, long_term_price = DataModel.predict_short_long_term_prices(best_model, scaler, latest_data)

    print(f'Short-term Prediction (7 days later): {short_term_price}')
    print(f'Long-term Prediction (30 days later): {long_term_price}')

    # Latest prices
    latest_data = data.loc[data['Ticker'] == tickers[0]].iloc[-1]
    print(f'Latest Prices:')
    print(f'Open Price: {latest_data["Open"]}')
    print(f'Close Price: {latest_data["Close"]}')
    print(f'High Price: {latest_data["High"]}')
    print(f'Low Price: {latest_data["Low"]}')


if __name__ == "__main__":
    main()
