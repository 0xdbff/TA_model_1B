import pandas as pd
from typing import List, Tuple
from datatypes import TOHLCVT
import re


def build_pd_df(lst: List[TOHLCVT], tp) -> pd.DataFrame:
    df = pd.DataFrame(lst)
    df.set_index("Index", inplace=True)

    last_row: None | TOHLCVT = None
    c = 0
    for row in df.itertuples():
        row = TOHLCVT(*row)
        if not last_row:
            c += 1
            last_row = row
            continue

        if row.Index - last_row.Index != (60 * tp):
            raise ValueError(
                f"Invalid time difference between rows! {row.Index - last_row.Index } seconds!= {60 * tp} current row: {c}"
            )

        c += 1
        last_row = row

    return df


def get(
    filename: str,
    find_tp_with_regex: bool = True,
    timeperiod: int | None = None,
    timestamp_len: int = 10,
) -> Tuple[List[List[TOHLCVT]], pd.DataFrame]:
    if find_tp_with_regex:
        pattern = r"_(\d+)\.csv$"
        match = re.search(pattern, filename)

        timeperiod = int(match.group(1)) if match else None
        if not timeperiod:
            raise ValueError("Timeperiod not found with regex!")
        return __get(filename, timeperiod, timestamp_len)

    if not timeperiod:
        raise ValueError("Timeperiod is required when not searching by regex!")
    return __get(filename, timeperiod, timestamp_len)


def __get(
    filename: str, timeperiod: int, timestamp_len: int = 10
) -> Tuple[List[List[TOHLCVT]], pd.DataFrame]:
    if timeperiod not in (1, 5, 15, 30, 60, 240, 720, 1440):
        raise ValueError("Invalid timeperiod!")

    df = pd.read_csv(filename, index_col=0)
    df.index.name = "timestamp"
    df.columns = ["O", "H", "L", "C", "V", "T"]

    first_timestamp = df.index[0]
    timestamp_len_current = len(str(int(first_timestamp)))

    if timestamp_len_current != 10:
        if timestamp_len_current == 13:
            df.index = (df.index.values // 1_000).astype(int)
        elif timestamp_len_current == 16:
            df.index = (df.index.values // 1_000_000).astype(int)
        elif timestamp_len_current == 19:
            df.index = (df.index.values // 1_000_000_000).astype(int)
        else:
            raise ValueError(
                f"Unexpected timestamp length: {timestamp_len_current} digits."
            )

    last_row = None
    last_g_row = None

    groups: List[List[TOHLCVT]] = []
    group_df: List[TOHLCVT] = []
    complete_data: List[TOHLCVT] = []

    for row in df.itertuples():
        row = TOHLCVT(*row)

        if len(str(row.Index)) != 10:
            raise ValueError("Invalid Timestamp (seconds)")

        if not last_row:
            last_row = row

            if not last_g_row:
                continue

            time_diff = row.Index - last_g_row.Index
            number_of_missing_rows = int(time_diff / (60 * timeperiod))
            for i in range(1, number_of_missing_rows):
                ts = last_g_row.Index + (i * 60 * timeperiod)

                r = TOHLCVT(
                    ts,
                    last_g_row.C,
                    last_g_row.C,
                    last_g_row.C,
                    last_g_row.C,
                    0,
                    0,
                    is_synthetic=True,
                )
                complete_data.append(r)
            complete_data.append(row)
            group_df.append(row)

            continue

        time_diff = row.Index - last_row.Index

        if time_diff % 60 != 0 or time_diff == 0:
            raise ValueError("Invalid time difference between rows!")

        if time_diff >= (600 * timeperiod):
            groups.append(group_df)
            group_df = []
            number_of_missing_rows = int(time_diff / (60 * timeperiod))
            for i in range(1, number_of_missing_rows):
                ts = last_row.Index + (i * 60 * timeperiod)

                r = TOHLCVT(
                    ts,
                    last_row.C,
                    last_row.C,
                    last_row.C,
                    last_row.C,
                    0,
                    0,
                    is_synthetic=True,
                )
                complete_data.append(r)
            complete_data.append(row)
            last_row = None
            last_g_row = row
            continue

        if time_diff == (60 * timeperiod):
            group_df.append(row)
            complete_data.append(row)
            last_row = row
            continue

        number_of_missing_rows = int(time_diff / (60 * timeperiod))

        for i in range(1, number_of_missing_rows):
            ts = last_row.Index + (i * 60 * timeperiod)

            r = TOHLCVT(
                ts,
                last_row.C,
                last_row.C,
                last_row.C,
                last_row.C,
                0,
                0,
                is_synthetic=True,
            )
            group_df.append(r)
            complete_data.append(r)

        group_df.append(row)
        complete_data.append(row)

        last_row = row
    groups.append(group_df)

    groups = [g for g in groups if len(g) >= 160]
    for group in groups:
        last_value = None
        for val in group:
            if last_value:
                if (val.Index - last_value.Index) != 60 * timeperiod:
                    raise ValueError(
                        f"Invalid time difference between rows! {val.Index - last_value.Index} seconds"
                    )
            last_value = val

    complete_df = build_pd_df(complete_data, timeperiod)

    return groups, complete_df


# [ [ [ 0101.0012 U 0114.2333 U 0099.8301 D 0112.9754 U 0097.4435 D EMA1_5 0112.2238 SMA1_5 0113.2389 V_55 T_32] [] ] ]

# 5 blocks use 53 tokens
# 46 * 32 = 1472 tokens

# Uptrend/ downtrend
# 1 * 32 # 32
# 16
# 8
# 4
# 2 # 62

# U_G5
# D_G5
# U_G10
# D_G10
# ...


# rsi
# 2 * 16 # 32
# 2 * 8  # 16
# 2 * 4  # 8
# 2 * 2  # 4
# 2 * 1  # 2 # 62

# SMA + EMA
# 6 * 16 * 2 # 192
# 6 * 8 * 2 # 96
# 6 * 4 * 2 # 48
# 6 * 2 * 2 # 24
# 6 * 1 * 2 # 12 # 372

# 42 Tokens end

# Last day change
# Last week change
# Last month change
# Last quarter change
# Last 6 months change
# Last year change

# 1472 + 16 + 62 + 372 + 36 + 62 = 2020
# tokens per sequence of 160 elements 5 * 32 # 2 hours 40min
# 28 tokens left, withouth including special tokens like bos, eos, ...

# (for finetuning)
# 6 prediction levels P1_blocktime -> predict 160/32 = 5 (price for the next 5nd element)
# 6 prediction levels P2_blocktime -> predict 160/16 = 10 nd element
# 6 prediction levels P3_blocktime -> predict 160/8 = 20
# 6 prediction levels P4_blocktime -> predict 160/4 = 40
# 6 prediction levels P5_blocktime -> predict 160/2 = 80
# 6 prediction levels P6_blocktime -> predict 160/1 = 160

# <<prediction>> P1_720 <eos> # 0118.0238

# every 10 elements, compute rsi10
# every 20 elements, compute rsi20
# every 20 elements, compute macd
# every 20 elements, compute SMA1_20
# every 20 elements, compute EMA1_20
# every 40 elements, compute SMA1_40
# every 40 elements, compute EMA1_40
# every 80 elements, compute SMA1_80
# every 80 elements, compute EMA1_80
# every 160 elements, compute SMA1_160
# every 160 elements, compute EMA1_160

# 5 10 20 40 80 160

# <<< LDC 0133.0923 LWC 0164.0345 LMC 0095.0483 LQC 0095.0483 L6MC 0095.0483 LYC 0095.0483 >>>
# 36 tokens

# in_range_groups = []
#
# for group in groups:
#     if len(group) < 150:
#         continue  # Ignore groups with less than 150 elements
#
#     # Split the group into subgroups if it has more than 300 elements
#     if len(group) > 300:
#         start = 0
#         while start < len(group):
#             end = min(start + 300, len(group))
#             subgroup = group[start:end]
#             if len(subgroup) >= 150:
#                 in_range_groups.append(subgroup)
#             start += 300
#     else:
#         # If the group is between 150 and 300 elements, just add it
#         in_range_groups.append(group)
#
# # Now iterate over in_range_groups and validate time differences
# count = 0
# for group in in_range_groups:
#     last_value = None
#     for val in group:
#         if last_value:
#             if val.Index - last_value.Index != 60:
#                 raise ValueError(
#                     f"Invalid time difference between rows! {val.Index - last_value.Index} seconds"
#                 )
#             count += 1
#         last_value = val


# 13 * 300 = 3900

# df["time_diff"] = df.index.to_series().diff().fillna(pd.Timedelta(seconds=0))
#
# # Define the interval range (1-4 minutes)
# min_interval = pd.Timedelta(minutes=1)
# max_interval = pd.Timedelta(minutes=10)
#
# # Create a boolean mask for intervals within the specified range
# df["within_interval"] = (df["time_diff"] >= min_interval) & (
#     df["time_diff"] <= max_interval
# )
#
# # Identify the contiguous ranges
# df["group"] = df["within_interval"].ne(df["within_interval"].shift()).cumsum()
#
# # Filter out the groups where intervals are not within the specified range
# valid_groups = df[df["within_interval"]].groupby("group").filter(lambda x: len(x) > 1)
#
# # Remove groups that have less than 720 elements
# large_groups = valid_groups.groupby("group").filter(lambda x: len(x) >= 720)
#
# # Drop helper columns (can be uncommented if needed)
# large_groups = large_groups.drop(columns=["time_diff", "within_interval"])
#
# groups = large_groups.groupby("group")
#
# print(groups.head(80))

# print(len(groups))
#
# for g in groups:
#     print(len(g))


# def resample_in_batches(
#     dataframe, batch_size="1D"
# ):  # '30D' for 30 days; adjust based on your data's distribution
#     start_date = dataframe.index.min()
#     end_date = dataframe.index.max()
#
#     result_frames = []
#     current_start = start_date
#
#     while current_start <= end_date:
#         current_end = min(
#             current_start + pd.Timedelta(batch_size) - pd.Timedelta(seconds=1), end_date
#         )
#         batch = dataframe[current_start:current_end]
#
#         resampled = batch.resample("s").asfreq()
#         resampled[["O", "H", "L", "C"]] = resampled[["O", "H", "L", "C"]].ffill()
#         resampled[["V", "T"]] = resampled[["V", "T"]].fillna(0)
#
#         result_frames.append(resampled)
#         current_start = current_end + pd.Timedelta(seconds=1)
#
#     return pd.concat(result_frames)
#
#
# # Apply the batch resampling
# df_resampled = resample_in_batches(df)
#
# # Reset the index if needed
# df_filled = df_resampled.reset_index()
#
# # Output the processed data
# print(df_filled.head(80))


# df_resampled = df.resample('1S').asfreq()
#
# # Fill OHLC with the last seen value (forward fill)
# df_resampled[['O', 'H', 'L', 'C']] = df_resampled[['O', 'H', 'L', 'C']].ffill()
#
# # Fill V and T with 0 for missing values
# df_resampled[['V', 'T']] = df_resampled[['V', 'T']].fillna(0)
#
# # Reset the index if needed
# df_filled = df_resampled.reset_index()

# this needs to be parallelized

# for each dataframe
# for name, df_group in dfs.items():
#     print(f"DataFrame for group {name}:")
#     print(df_group)
#     print("\n")
#
#     # create a data item for each subgroup (context for the input sequence)
#
#     # first find how many subgroups to segment in the current group of >=720 elements
#
#     # fill blanks (add items)
#
#     complete_df = df_group  # clone
#
#     for item in df_group:
#         # identify missing information, and add default! (will increase the number of elements in the df_group)
#
#         # complete_df.add()
#         pass
#
#     number_of_subgroups = int(len(df_group) / 720)
#
#     # this needs to be parallelized!
#
#     for df_subgroup in range(number_of_subgroups):
#
#         input_sequence: str = ""
#
#         # possible quadratic time complexity..
#
#         for item in df_subgroup:
#
#             # update the standardazation limits for this asset type
#
#             # compute relative price variations
#
#             # compute standardized Technical indicators (SMA, RSI, MACD etc.)
#
#             input_sequence.join()
#
#             pass
