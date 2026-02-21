import os
import re
import math
import pandas as pd

from typing import List
from datatypes import TOHLCVT, TimePeriod
from collections import defaultdict
import argparse

from all import get


def get_filepaths():

    directory_path = "/home/db/data"
    pattern = r"(\w+)_1.csv$"

    size_threshold = 50 * 1024 * 1024

    extracted_names = []
    total_size = 0

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if re.match(pattern, file_name):
                file_path = os.path.join(root, file_name)
                if os.path.getsize(file_path) > size_threshold:
                    match = re.match(pattern, file_name)
                    if match:
                        total_size += os.path.getsize(file_path)
                        extracted_name = match.group(1)
                        extracted_names.append(extracted_name)

    file_paths_dict = defaultdict(list)

    paths_complete_pairs = []

    for name in extracted_names:
        pattern = rf"{name}_(\d+)\.csv$"

        for root, _, files in os.walk(directory_path):
            for file_name in files:
                match = re.match(pattern, file_name)
                if match:
                    pair_name = match.group(1)
                    file_path = os.path.join(root, file_name)
                    file_paths_dict[pair_name].append(file_path)

    for timeperiod, paths in file_paths_dict.items():
        if len(paths) <= 8:
            continue

        for path in paths:
            pattern = r"(\w+)_(\d+).csv$"

            match = re.search(pattern, path)
            if not match:
                raise ValueError("Invalid filename!")
            complete_df_name = match.group(1) + f"_{timeperiod}.csv"

            paths_complete_pairs.append((path, int(timeperiod), complete_df_name))

    return paths_complete_pairs


class TaParser:
    r""" """

    sequence: str | None = None
    __complete_df: pd.DataFrame | None = None
    __data: List[TOHLCVT] | None = None

    __volume_mean: float | None = None
    __volume_std_dev: float | None = None
    __trades_mean: float | None = None
    __trades_std_dev: float | None = None

    def __init__(
        self,
        data: List[TOHLCVT],
        complete_df: pd.DataFrame | None = None,
        period: TimePeriod = TimePeriod._1m,
        length: int = 160,
    ):
        """ """
        if (
            length % 5 != 0
            or ((length // 5) & ((length // 5) - 1)) != 0
            or length < 20
            or len(data) != length
        ):
            raise ValueError(
                "Invalid length for input sequence!, needs to be 5 * 2**n, n > 1"
            )

        self.length = length
        self.period = period
        self.__data = data
        self.__complete_df = complete_df

        self.__compute_volume_mean()
        self.__compute_trades_mean()

        self.__compute_volume_std_dev()
        self.__compute_trades_std_dev()

        self.__process()

    def __call__(self):
        return self.sequence

    def __process(self):
        sequence = "{ "
        if not self.__data:
            raise ValueError("No data to process!")

        last_record: TOHLCVT | None = None
        for index, record in enumerate(self.__data, start=1):
            if last_record and record.Index - last_record.Index != self.period.value:
                raise ValueError("Invalid time difference between rows!")

            if last_record:
                sequence = (
                    sequence
                    + self.__encode_value_change(last_record, record)
                    + " "
                    + self.__encode_standardized_volume(record.V)
                    + " "
                    + self.__encode_standardized_trades(record.T)
                    + " "
                )

            if index % 5 == 0:
                window_records: List[TOHLCVT] = []
                if index % 2 != 0:
                    window_records = self.__data[index - 5 : index]
                    relative_sma = self.__encode_relative_sma_and_group_trend(
                        window_records, 5
                    )
                    relative_ema = self.__encode_relative_ema(window_records, 5)
                    rsi = self.__encode_rsi(window_records, 5)

                    sequence = (
                        sequence + f"| {relative_ema} {relative_sma} {rsi} " + "} { "
                    )
                    last_record = record
                    continue

                n = int(math.log2(index / 5)) + 1

                for ta_level in range(0, n):
                    window_size = 5 * 2**ta_level
                    window_records = self.__data[index - window_size : index]

                    if index % window_size != 0:
                        continue

                    relative_sma = self.__encode_relative_sma_and_group_trend(
                        window_records, window_size
                    )
                    relative_ema = self.__encode_relative_ema(
                        window_records, window_size
                    )
                    rsi = self.__encode_rsi(window_records, window_size)

                    sequence = (
                        sequence + f"| {relative_ema} {relative_sma} {rsi} "
                        if window_size != 5
                        else sequence
                        + f"| {relative_ema} {relative_sma} {rsi} {
                            self.__encode_standardized_mean_volume(window_records, window_size)} {
                            self.__encode_standardized_mean_trades(window_records, window_size)} "
                    )

                if index < self.length:
                    sequence = sequence + "} { "

            last_record = record

        sequence = sequence + self.__encode_historical_values()

        self.sequence = sequence

    def __standardize_scaled(
        self, value: float, mean: float, std_dev: float, scaling_factor: float = 16.67
    ) -> int:
        """ """
        z_score = (value - mean) / std_dev
        scaled_value = 50 + (z_score * scaling_factor)
        return int(round(max(0, min(100, scaled_value))))

    def __compute_volume_mean(self):
        """Computes the mean of volumes in the data."""
        if not self.__data:
            raise ValueError("No data to process!")

        total_volume = sum(item.V for item in self.__data)
        self.__volume_mean = total_volume / len(self.__data)

    def __compute_volume_std_dev(self):
        """Computes the standard deviation of volumes in the data."""
        if not self.__data:
            raise ValueError("No data to process!")

        if len(self.__data) < 2:
            raise ValueError(
                "At least two data points are required to compute standard deviation."
            )

        if not self.__volume_mean:
            raise ValueError("Volume mean is not computed!")

        variance = sum((item.V - self.__volume_mean) ** 2 for item in self.__data) / (
            len(self.__data) - 1
        )
        self.__volume_std_dev = math.sqrt(variance)

    def __compute_trades_mean(self):
        """Computes the mean of trades in the data."""
        if not self.__data:
            raise ValueError("No data to process!")

        total_trades = sum(item.T for item in self.__data)
        self.__trades_mean = total_trades / len(self.__data)

    def __compute_trades_std_dev(self):
        """Computes the standard deviation of trades in the data."""
        if not self.__data:
            raise ValueError("No data to process!")

        if len(self.__data) < 2:
            raise ValueError(
                "At least two data points are required to compute standard deviation."
            )

        if not self.__trades_mean:
            raise ValueError("Volume mean is not computed!")

        variance = sum((item.T - self.__trades_mean) ** 2 for item in self.__data) / (
            len(self.__data) - 1
        )
        self.__trades_std_dev = math.sqrt(variance)

    def __scale_non_linear(
        self,
        percentage,
        scale_limit: int = 2**16 - 1,
        max_supported_diff: float = 10000,
        inverse_exponential_scale: float = 0.3,
    ) -> int:
        """
        - prev_idx : `int` - The index of the previous record.
        - idx : `int` - The index of the current record.
        - `std_range_limit` : `int` - The maximum value of the relative
            standardized value.
        - `max_supported_diff` : `float` - The maximum supported relative difference
            between the Close values.
        - `inverse_exponential_scale` : `float` - The inverse exponential scale
        is_positive = percentage >= 0
        """
        is_positive = percentage >= 0

        if abs(percentage) > max_supported_diff:
            return scale_limit if is_positive else -scale_limit

        scaled_relative_change = int(
            round(
                ((abs(percentage) / max_supported_diff) ** inverse_exponential_scale)
                * scale_limit
            )
        )

        return scaled_relative_change if is_positive else -scaled_relative_change

    def __check_index_range(self, start_idx: int, end_idx: int):
        """ """
        if not self.__data:
            raise ValueError("No data to process!")

        start_idx = start_idx if start_idx >= 0 else len(self.__data) + start_idx
        end_idx = end_idx if end_idx >= 0 else len(self.__data) + end_idx

        if (
            not all(0 <= i < len(self.__data) for i in (start_idx, end_idx))
            and start_idx >= end_idx
        ):
            raise ValueError("Invalid index values!")

    def __encode_hex(self, value: int) -> str:
        sign = "-" if value < 0 else "+"

        hex = f"{(abs(value)):04x}"

        return f"{sign} {hex[:2]} {hex[2:]}"

    def __encode_trend(self, v: int | float) -> str:
        if v > 0:
            return "U"
        elif v < 0:
            return "D"
        else:
            return "N"

    def __encode_value_change(
        self,
        last_record: TOHLCVT,
        record: TOHLCVT,
    ) -> str:
        """
        Convert two records of TOHLCVT (only Close value is used) to a relative
        standardized value representing the value change, following a non-linear
        scaling formula.

        # Returns

        - `int` : The relative standardized value of the Close value change for the
        current index, from -'self.std_range_limit' to 'self.std_range_limit'.
        """
        if last_record.C == 0:
            raise ZeroDivisionError("Last value cannot be zero!")

        relative_change = (record.C - last_record.C) / last_record.C
        scaled_value = self.__scale_non_linear(relative_change)

        trend = self.__encode_trend(scaled_value)

        return f"{self.__encode_hex(scaled_value)} {trend}"

    def __encode_standardized_volume(self, value: float) -> str:
        """ """
        if not self.__volume_mean or not self.__volume_std_dev:
            raise ValueError("Volume mean or standard deviation not computed!")

        value = self.__standardize_scaled(
            value, self.__volume_mean, self.__volume_std_dev
        )
        return f"V_{value}"

    def __encode_standardized_trades(self, value: float) -> str:
        """ """
        if not self.__trades_mean or not self.__trades_std_dev:
            raise ValueError("Trades mean or standard deviation not computed!")

        value = self.__standardize_scaled(
            value, self.__trades_mean, self.__trades_std_dev
        )
        return f"T_{value}"

    def __encode_standardized_mean_volume(
        self, window_records: List[TOHLCVT], window_size: int
    ) -> str:
        """ """
        if not self.__volume_mean or not self.__volume_std_dev:
            raise ValueError("Volume mean or standard deviation not computed!")

        if window_size != len(window_records):
            raise ValueError("Invalid length of window records!")

        mean = sum(r.V for r in window_records) / window_size
        standardized_mean = self.__standardize_scaled(
            mean, self.__volume_mean, self.__volume_std_dev
        )
        return f"V_G{window_size}_{standardized_mean}"

    def __encode_standardized_mean_trades(
        self, window_records: List[TOHLCVT], window_size: int
    ) -> str:
        """ """
        if not self.__trades_mean or not self.__trades_std_dev:
            raise ValueError("Trades mean or standard deviation not computed!")

        if window_size != len(window_records):
            raise ValueError("Invalid length of window records!")

        mean = sum(r.T for r in window_records) / window_size
        standardized_mean = self.__standardize_scaled(
            mean, self.__trades_mean, self.__trades_std_dev
        )
        return f"T_G{window_size}_{standardized_mean}"

    def __encode_relative_ema(
        self, window_records: List[TOHLCVT], window_size: int
    ) -> str:
        """ """
        if window_size != len(window_records):
            raise ValueError("Invalid length of window records!")

        close_prices = [r.C for r in window_records]

        ema = close_prices[0]
        alpha = 2 / (len(close_prices) + 1)

        for price in close_prices[1:]:
            ema = ema + alpha * (price - ema)

        relative_change = (ema - close_prices[0]) / close_prices[0]
        scaled_value = self.__scale_non_linear(relative_change)

        return f"EMA{window_size}_{self.period.value} {self.__encode_hex(scaled_value)}"

    def __encode_relative_sma_and_group_trend(
        self, window_records: List[TOHLCVT], window_size: int
    ) -> str:
        """ """
        if window_size != len(window_records):
            raise ValueError(
                f"Invalid length of window records!{window_size}!={len(window_records)}"
            )

        close_prices = [r.C for r in window_records]
        sma = sum(close_prices) / len(close_prices)

        relative_change = (sma - close_prices[0]) / close_prices[0]
        scaled_value = self.__scale_non_linear(relative_change)

        return f"SMA{window_size}_{self.period.value} {self.__encode_hex(scaled_value)} {
            self.__encode_group_trend(scaled_value, window_size)}"

    def __encode_rsi(self, window_records: List[TOHLCVT], window_size: int) -> str:
        """ """
        if window_size != len(window_records):
            raise ValueError("Invalid length of window records!")

        changes = [
            window_records[i].C - window_records[i - 1].C
            for i in range(1, len(window_records))
        ]

        gains = [change if change > 0 else 0 for change in changes]
        losses = [-change if change < 0 else 0 for change in changes]

        average_gain = sum(gains) / len(gains)
        average_loss = sum(losses) / len(losses)

        rs = float("inf") if average_loss == 0 else average_gain / average_loss
        rsi = 100 if rs == float("inf") else 100 - (100 / (1 + rs))
        rsi = int(round(rsi))

        if average_gain == 0 and average_loss == 0:
            rsi = 50

        return f"RSI{window_size}_{self.period.value} r{rsi}"

    def __encode_group_trend(self, v: float, window: int) -> str:
        """ """
        trend = self.__encode_trend(v)
        return f"{trend}_G{window}"

    def __encode_historical_values(self) -> str:
        if not self.__data:
            raise ValueError("No data to process!")

        if self.__complete_df is None:
            raise ValueError("Complete data not available!")

        latest_record = self.__data[-1].Index

        try:
            last_day_close = self.__complete_df.loc[latest_record - 86400, "C"]
            last_week_change = self.__complete_df.loc[latest_record - 604800, "C"]
            last_month_change = self.__complete_df.loc[latest_record - 2592000, "C"]
            last_quarter_change = self.__complete_df.loc[latest_record - 7776000, "C"]
            last_6months_close = self.__complete_df.loc[latest_record - 15552000, "C"]
            last_year_close = self.__complete_df.loc[latest_record - 31536000, "C"]

            current_close = self.__data[-1].C

            last_day_change = self.__scale_non_linear(
                (current_close - last_day_close) / last_day_close
            )
            last_week_change = self.__scale_non_linear(
                (current_close - last_week_change) / last_week_change
            )
            last_month_change = self.__scale_non_linear(
                (current_close - last_month_change) / last_month_change
            )
            last_quarter_change = self.__scale_non_linear(
                (current_close - last_quarter_change) / last_quarter_change
            )
            last_6months_change = self.__scale_non_linear(
                (current_close - last_6months_close) / last_6months_close
            )
            last_year_change = self.__scale_non_linear(
                (current_close - last_year_close) / last_year_close
            )

        except Exception as _:
            return "<<< NaN >>>"

        return (
            "} <<<"
            + f"LDC {self.__encode_hex(last_day_change)} "
            + f"LWC {self.__encode_hex(last_week_change)} "
            + f"LMC {self.__encode_hex(last_month_change)} "
            + f"LQC {self.__encode_hex(last_quarter_change)} "
            + f"L6MC {self.__encode_hex(last_6months_change)} "
            + f"LYC {self.__encode_hex(last_year_change)} "
            + ">>>"
        )


# def process_sequence(sequence, timeperiod, complete_df):
#     timeperiod = TimePeriod(timeperiod * 60)
#     parser = TaParser(sequence, period=timeperiod, complete_df=complete_df)
#
#     return parser()


def run(sequences_dataset, tp):
    total = len(sequences_dataset)
    nan_count = 0

    # with ProcessPoolExecutor(max_workers=12) as executor:
    #     loop = asyncio.get_running_loop()
    #
    #     tasks = [
    #         loop.run_in_executor(
    #             executor, process_sequence, sequence, period, complete_df
    #         )
    #         for count, (sequence, period, complete_df) in enumerate(
    #             sequences_dataset, start=1
    #         )
    #     ]
    #
    #     results = await asyncio.gather(*tasks)

    parent_dir = "/home/db/dev/TaSystem/data/pretrain/v1-small"

    with open(f"{parent_dir}/dataset_{tp}.pretrain", "a") as file:
        for s, p, c in sequences_dataset:
            p = TimePeriod(p * 60)
            parser = TaParser(s, period=p, complete_df=c)
            result = parser()
            del parser
            if type(result) == str:
                file.write(str(result) + "\n")
                if "NaN" in result:
                    nan_count += 1
            else:
                raise ValueError("Invalid result!")

    print(f"Total NaN count: {nan_count} out of {total}")


# async def process_sequence(sem, sequence, count, total):
#     async with sem:
#         print(f"Executing group {count} out of {total}")
#         # Offload the blocking operation to a separate thread
#         obj = await asyncio.to_thread(TaParser, sequence)
#         # print(parser())
#
#
# async def main(sequences_dataset):
#     # Semaphore to limit the number of concurrent threads to 12
#     sem = asyncio.Semaphore(12)
#     total = len(sequences_dataset)
#     tasks = []
#
#     # Create tasks for processing each sequence in the dataset
#     for count, sequence in enumerate(sequences_dataset, start=1):
#         task = process_sequence(sem, sequence, count, total)
#         tasks.append(task)
#
#     # Run all tasks concurrently and wait for them to complete
#     await asyncio.gather(*tasks)


# async def process_sequence(sem, sequence, count, total):
#     async with sem:
#         print(f"Executing group {count} out of {total}")
#         parser = TaParser(sequence)
#         result = await asyncio.to_thread(parser)
#         print(result)
#
#
# async def main(sequences_dataset):
#     sem = asyncio.Semaphore(12)
#     total = len(sequences_dataset)
#     tasks = []
#
#     for count, sequence in enumerate(sequences_dataset, start=1):
#         tasks.append(process_sequence(sem, sequence, count, total))
#
#     await asyncio.gather(*tasks)


def main(timeperiod: int = 1):

    total = 0
    timeperiods = {1, 5, 15, 30, 60, 240, 720, 1440}

    if timeperiod not in timeperiods:
        raise ValueError("Invalid timeperiod!")

    iteration = 0
    files = get_filepaths()

    for file, tp, _ in files:
        if tp != timeperiod:
            continue
        iteration += 1
        sequences_dataset = []
        print(f"Processing file: {file}, {iteration} out of {len(files)}")
        groups, complete_df = get(file)
        for g in groups:
            l = len(g)
            step = 1
            if timeperiod == 1:
                step = 160
            elif timeperiod == 5:
                step = 40
            elif timeperiod == 15:
                step = 18
            elif timeperiod == 30:
                step = 9
            elif timeperiod == 60:
                step = 6
            elif timeperiod == 240:
                step = 4

            for sb in range(0, l, step):
                if sb + 160 < l:
                    sequence: List[TOHLCVT] = g[sb : sb + 160]
                    sequences_dataset.append((sequence, timeperiod, complete_df))

                if len(sequence) != 160:
                    raise ValueError("Invalid length of group!")

        print("Total number of sequences to process:", len(sequences_dataset))
        print("Processing sequences...")
        total += len(sequences_dataset)

        run(sequences_dataset, timeperiod)
        del groups, complete_df, sequences_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a time period for data analysis."
    )
    parser.add_argument(
        "timeperiod",
        type=int,
        nargs="?",
        default=1,
        help="Time period to process (must be one of: 1, 5, 15, 30, 60, 240, 720, 1440).",
    )

    args = parser.parse_args()
    main(args.timeperiod)


def reverse_scale_non_linear(
    scaled_value: int,
    scale_limit: int = 2**16 - 1,
    max_supported_diff: float = 10000,
    inverse_exponential_scale: float = 0.3,
) -> float:
    """
    Reverse the scaling to get the original percentage.
    """
    is_positive = scaled_value >= 0

    if abs(scaled_value) == scale_limit:
        return max_supported_diff if is_positive else -max_supported_diff

    original_value = max_supported_diff * (
        (abs(scaled_value) / scale_limit) ** (1 / inverse_exponential_scale)
    )

    return original_value if is_positive else -original_value
