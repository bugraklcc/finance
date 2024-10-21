import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class DataModel:
    @staticmethod
    def prepare_data(data):
        data.dropna(inplace=True)
        X = data[['SMA_10', 'SMA_50', 'Return', 'Volume', 'SMA_20', 'Volatility', 'Momentum']]
        y = data['Close'].shift(-1).dropna()
        X = X[:-1]
        return X, y

    @staticmethod
    def train_model(X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')

        return best_model, scaler

    @staticmethod
    def predict_future_price(model, scaler, latest_data, days_ahead):
        future_date = datetime.now() + timedelta(days=days_ahead)
        latest_data['FutureDate'] = future_date
        latest_features = pd.DataFrame({
            'SMA_10': [latest_data['SMA_10']],
            'SMA_50': [latest_data['SMA_50']],
            'Return': [latest_data['Return']],
            'Volume': [latest_data['Volume']],
            'SMA_20': [latest_data['SMA_20']],
            'Volatility': [latest_data['Volatility']],
            'Momentum': [latest_data['Momentum']]
        })

        latest_features_scaled = scaler.transform(latest_features)
        future_price = model.predict(latest_features_scaled)
        return future_price[0]

    @staticmethod
    def predict_short_long_term_prices(model, scaler, latest_data):
        future_prices_short = DataModel.predict_future_price(model, scaler, latest_data, days_ahead=7)
        future_prices_long = DataModel.predict_future_price(model, scaler, latest_data, days_ahead=30)
        return future_prices_short, future_prices_long
