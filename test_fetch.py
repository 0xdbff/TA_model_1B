from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import NamedTuple
import time


# Define the TOHLCVT NamedTuple
class TOHLCVT(NamedTuple):
    """
    `TOHLCVT` NamedTuple that represents a financial data record with Open,
    High, Low, Close, Volume, Trades indexed by Timestamps.

    ## Attributes

    - **Index** : `int` - The index of the data record, timestamp identifying
        the time (10-digit unix timestamp in seconds).
    - **O** : `float` - The opening price of the asset.
    - **H** : `float` - The highest price of the asset during the time period.
    - **L** : `float` - The lowest price of the asset during the time period.
    - **C** : `float` - The closing price of the asset.
    - **V** : `float` - The trading volume of the asset during the time period.
    - **T** : `int` - The number of finished trades during the time period.
    - **is_synthetic** : `bool` - Indicates if the data is synthetic (default False).
    """

    Index: int
    O: float
    H: float
    L: float
    C: float
    V: float
    T: int
    is_synthetic: bool = False


# Initialize the Binance client (no API key needed for public data)
client = Client()

# Map the time periods in minutes to Binance interval strings
interval_mapping = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    240: "4h",
    720: "12h",
    1440: "1d",
}


# Function to fetch TOHLCVT data and closing prices
def get_tohlcvt_data(symbol: str, time_period: int, get_historical: bool = True):
    """
    Fetches the last 160 TOHLCVT data records for the given symbol and time period.
    Also fetches the closing prices for specific past periods (cached).

    Parameters:
    - symbol: The trading pair symbol, e.g., 'BTCUSDT'.
    - time_period: The time period in minutes, one of [1, 5, 15, 30, 60, 240, 720, 1440].

    Returns:
    - data_list: A list of TOHLCVT namedtuples.
    - closing_prices: A dictionary containing the closing prices for specific past periods.
    """

    # Get the Binance interval string
    interval = interval_mapping.get(time_period)
    if not interval:
        raise ValueError(
            f"Invalid time period: {time_period}. Must be one of {list(interval_mapping.keys())}."
        )

    # Fetch the last 160 klines
    klines = client.get_klines(symbol=symbol, interval=interval, limit=161)
    data_list = []
    for kline in klines:
        open_time = int(kline[0] // 1000)  # Convert milliseconds to seconds
        open_price = float(kline[1])
        high_price = float(kline[2])
        low_price = float(kline[3])
        close_price = float(kline[4])
        volume = float(kline[5])
        number_of_trades = int(kline[8])
        data_list.append(
            TOHLCVT(
                Index=open_time,
                O=open_price,
                H=high_price,
                L=low_price,
                C=close_price,
                V=volume,
                T=number_of_trades,
            )
        )

    # Fetch daily candlestick data for the past year for closing prices
    today = datetime.utcnow()
    start_date = today - timedelta(days=366)  # Extra day to ensure we have data
    klines_daily = client.get_historical_klines(
        symbol,
        Client.KLINE_INTERVAL_1DAY,
        start_str=start_date.strftime("%d %b %Y %H:%M:%S"),
        end_str=today.strftime("%d %b %Y %H:%M:%S"),
    )

    if get_historical:

        # Process daily klines into a DataFrame
        df_daily = pd.DataFrame(
            klines_daily,
            columns=[
                "OpenTime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "CloseTime",
                "QuoteAssetVolume",
                "NumberOfTrades",
                "TakerBuyBaseAssetVolume",
                "TakerBuyQuoteAssetVolume",
                "Ignore",
            ],
        )
        df_daily["OpenTime"] = pd.to_datetime(df_daily["OpenTime"], unit="ms")
        df_daily["CloseTime"] = pd.to_datetime(df_daily["CloseTime"], unit="ms")
        df_daily["Close"] = df_daily["Close"].astype(float)

        # Define the target dates
        dates = {
            "Yesterday": today - timedelta(days=1),
            "Last Week": today - timedelta(weeks=1),
            "Last Month": today - relativedelta(months=1),
            "Last Quarter": today - relativedelta(months=3),
            "Last 6 Months": today - relativedelta(months=6),
            "Last Year": today - relativedelta(years=1),
        }

        # Fetch the closing prices for the specified dates
        closing_prices = {}
        for label, date in dates.items():
            # Find the closest date in df_daily on or before the target date
            df_filtered = df_daily[df_daily["OpenTime"] <= date]
            if not df_filtered.empty:
                last_row = df_filtered.iloc[-1]
                closing_price = last_row["Close"]
                closing_prices[label] = {
                    "Price": closing_price,
                    "Date": last_row["OpenTime"].date(),
                }
            else:
                closing_prices[label] = {"Price": None, "Date": None}

        return data_list, closing_prices
    else:
        return data_list, None


# Example usage:
if __name__ == "__main__":
    symbol = "BTCUSDT"  # You can change this to any symbol available on Binance
    time_period = 1  # For example, 5-minute intervals

    data_list, closing_prices = get_tohlcvt_data(symbol, time_period)

    # Print the closing prices
    print("\nClosing Prices:")
    for period, info in closing_prices.items():
        if info["Price"] is not None:
            print(f"{period} ({info['Date']}): {info['Price']}")
        else:
            print(f"{period}: No data available")

    # Print the last few entries of TOHLCVT data
    print(f"\nLast few entries for {symbol} at {time_period}-minute interval:")
    for record in data_list[-5:]:
        print(record)

    print(len(data_list))

    data_list, _ = get_tohlcvt_data(symbol, 5, get_historical=False)
    print(f"\nLast few entries for {symbol} at {time_period}-minute interval:")
    for record in data_list[-5:]:
        print(record)

    print(len(data_list))

    data_list, _ = get_tohlcvt_data(symbol, 15, get_historical=False)
    print(f"\nLast few entries for {symbol} at {time_period}-minute interval:")
    for record in data_list[-5:]:
        print(record)

    print(len(data_list))

    data_list, _ = get_tohlcvt_data(symbol, 30, get_historical=False)
    print(f"\nLast few entries for {symbol} at {time_period}-minute interval:")
    for record in data_list[-5:]:
        print(record)

    print(len(data_list))

    data_list, _ = get_tohlcvt_data(symbol, 60, get_historical=False)
    print(f"\nLast few entries for {symbol} at {time_period}-minute interval:")
    for record in data_list[-5:]:
        print(record)

    print(len(data_list))

    data_list, _ = get_tohlcvt_data(symbol, 240, get_historical=False)
    print(f"\nLast few entries for {symbol} at {time_period}-minute interval:")
    for record in data_list[-5:]:
        print(record)

    print(len(data_list))

    data_list, _ = get_tohlcvt_data(symbol, 720, get_historical=False)
    print(f"\nLast few entries for {symbol} at {time_period}-minute interval:")
    for record in data_list[-5:]:
        print(record)

    print(len(data_list))

    data_list, _ = get_tohlcvt_data(symbol, 1440, get_historical=False)
    print(f"\nLast few entries for {symbol} at {time_period}-minute interval:")
    for record in data_list[-5:]:
        print(record)

    print(len(data_list))
