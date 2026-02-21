from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
import time  # for sleep between requests if needed


def main(interval, client, tp):
    # Initialize the Binance Client

    # Define the symbol and time interval
    symbol = "BTCUSDT"
    # interval = Client.KLINE_INTERVAL_30MINUTE

    # Calculate the start and end dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=355 * 2)

    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    # Loop to fetch data in batches of 1000
    while start_date < end_date:
        # Fetch the data
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_str=start_date.strftime("%d %b %Y %H:%M:%S"),
            limit=1000,
        )

        if not klines:
            break  # Exit if no more data is returned

        # Convert to DataFrame and append to all_data
        temp_df = pd.DataFrame(
            klines,
            columns=[
                "Timestamp",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close Time",
                "Quote Asset Volume",
                "Number of Trades",
                "Taker Buy Base Asset Volume",
                "Taker Buy Quote Asset Volume",
                "Ignore",
            ],
        )

        # Select only the relevant columns and keep timestamp in Unix format
        temp_df = temp_df[
            ["Timestamp", "Open", "High", "Low", "Close", "Volume", "Number of Trades"]
        ]

        # Append to the full dataset
        all_data = pd.concat([all_data, temp_df], ignore_index=True)

        # Update start_date for the next loop to the last timestamp fetched
        last_timestamp = int(temp_df["Timestamp"].iloc[-1])  # Unix format
        start_date = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(
            minutes=1
        )

        # Optional: Pause briefly to respect API rate limits
        time.sleep(0.1)

    # Save to a CSV file
    all_data.to_csv(f"btc_{tp}.csv", index=False)

    # Display the first few rows of the full dataset

    print(all_data.head())


if __name__ == "__main__":
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_SECRET_KEY"
    client = Client(api_key, api_secret)

    klines = (
        (client.KLINE_INTERVAL_5MINUTE, 5),
        (client.KLINE_INTERVAL_15MINUTE, 15),
        (client.KLINE_INTERVAL_30MINUTE, 30),
        (client.KLINE_INTERVAL_1HOUR, 60),
        (client.KLINE_INTERVAL_4HOUR, 240),
        (client.KLINE_INTERVAL_12HOUR, 720),
        (client.KLINE_INTERVAL_1DAY, 1440),
    )

    for interval, tp in klines:
        main(interval, client, tp)
