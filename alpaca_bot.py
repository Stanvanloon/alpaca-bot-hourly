from os import getenv
from sys import exit, stderr
from joblib import load
from numpy import where
from datetime import datetime, timedelta
from alpaca_trade_api import REST
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

client = CryptoHistoricalDataClient()

API_KEY = getenv("ALPACA_API_KEY_ID")
API_SECRET = getenv("ALPACA_API_SECRET_KEY")
SYMBOL = 'BTC/USD'
BASE_URL = "https://paper-api.alpaca.markets"

if not API_KEY or not API_SECRET:
    print("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set in environment.", file=stderr)
    exit(1)

def main():
    api = REST(API_KEY, API_SECRET, BASE_URL)

    # Load per-trained model
    rf_loaded = load("Models/RF_20260129.joblib")

    # Get historical data
    now = datetime.now() - timedelta(hours=1)
    past = now - timedelta(hours=5)

    request_params = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Hour,
        start=datetime(past.year, past.month, past.day, past.hour),
        end=datetime(now.year, now.month, now.day, now.hour)
    )

    btc_bars = client.get_crypto_bars(request_params)
    btc_bars = btc_bars.df

    btc_bars['Result'] = where(btc_bars['close'] > btc_bars['open'], 1, 0)

    for i in range(5):
        btc_bars[f"hour_{i+1}"] = btc_bars['Result'].shift(i+1).astype('Int64')

    # Gather the only input data point
    btc_bars = btc_bars.dropna()
    input = btc_bars[[f"hour_{i+1}" for i in range(5)]]

    # Calculate the prediction for the current hour
    prediction = int(rf_loaded.predict(input)[0])

    # Get current position size
    total_position = 0.0
    for pos in api.list_positions():
        print(pos)
        if pos.symbol == 'BTCUSD':
            total_position += float(pos.qty)

    # Execute trades based on prediction
    if prediction == 1 and total_position == 0.0:
        api.submit_order(symbol=SYMBOL, qty=0.1, side='buy', type='market', time_in_force='gtc')
        print("Bought 0.1 BTC")
    elif prediction == 0 and total_position > 0.0:
        api.submit_order(symbol=SYMBOL, qty=total_position, side='sell', type='market', time_in_force='gtc')
        print(f"Sold {total_position} BTC")
    else:
        print("No action taken")

if __name__ == "__main__":
    main()