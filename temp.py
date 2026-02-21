import numpy as np
import math
import pandas as pd

from typing import List
from enum import Enum
from typing import NamedTuple


class TOHLCVT(NamedTuple):
    Index: int
    O: float
    H: float
    L: float
    C: float
    V: float
    T: int


class TimePeriod(Enum):
    _1m = 60
    _5m = 300
    _15m = 900
    _30m = 1800
    _1h = 3600
    _4h = 14400
    _12h = 43200
    _24h = 86400


class TaParser:

    sequence: str | None = None
    complete_df: pd.DataFrame | None = None
    data: List[TOHLCVT] | None = None

    def __init__(
        self,
        data: List[TOHLCVT],
        period: TimePeriod = TimePeriod._1m,
        length: int = 160,
    ):
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
        self.data = data
        self.period = period

        self.max_volume = 0
        self.min_volume = 0
        self.max_trades = 0
        self.min_trades = 0

        self.__process()
        print(self.sequence)

    def __process(self):
        sequence = "{ "
        if not self.data:
            raise ValueError("No data to process!")

        last_value: TOHLCVT | None = None
        for index, v in enumerate(self.data, start=1):
            if last_value and v.Index - last_value.Index != self.period.value:
                raise ValueError("Invalid time difference between rows!")

            # compute relative values, and U D
            sequence = sequence + "0101.0012 U "

            if index % 5 == 0:
                if index % 2 != 0:
                    relative_sma = self.__encode_relative_sma(5)
                    relative_ema = self.__encode_relative_ema(5)
                    rsi = self.__encode_rsi(5)
                    group_trend = self.__encode_group_trend(5)

                    sequence = (
                        sequence + f"| {relative_sma} {relative_ema} {rsi} {group_trend} " + "} { "
                    )
                    last_value = v
                    continue

                n = int(math.log2(index / 5)) + 1

                for ta_level in range(0, n):
                    window = 5 * 2**ta_level

                    if index % window != 0:
                        continue

                    relative_sma = self.__encode_relative_sma(window)
                    relative_ema = self.__encode_relative_ema(window)
                    rsi = self.__encode_rsi(window)
                    group_trend = self.__encode_group_trend(window)

                    sequence = (
                        sequence + f"| {relative_sma} {relative_ema} {rsi} {group_trend} "
                        if window != 5
                        else sequence
                        + f"| {relative_sma} {relative_ema} {rsi} {group_trend} {
                            self.__encode_standardized_volume()} {
                            self.__encode_standardized_trades()}"
                    )

                if index < self.length:
                    sequence = sequence + "} { "

            last_value = v

        sequence = sequence + self.__encode_historical_values()

        self.sequence = sequence

    def __convert_to_relative_values(self):
        pass

    def __encode_standardized_volume(self):
        pass

    def __encode_standardized_trades(self):
        pass

    def __encode_relative_ema(self, window=10, precision=4) -> str:
        return f"EMA{window}_{self.period.value} 0112.2238"

    def __encode_relative_sma(self, window=10, precision=4) -> str:
        return f"SMA{window}_{self.period.value} 0117.2238"

    def __encode_rsi(self, window=10) -> str:
        return f"RSI{window}_{self.period.value} r59"

    def __encode_group_trend(self, window=10) -> str:
        v = 1
        trend = "U" if v > 0 else "D"

        return f"{trend}_G{window}"

    def __encode_historical_values(self) -> str:
        return "} <<<LDC 0133.0923 LWC 0164.0345 LMC 0095.0483 LQC 0095.0483 L6MC 0095.0483 LYC 0095.0483>>>"


list = []

for i in range(160):
    ts = i * 60
    r = TOHLCVT(
        ts,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    list.append(r)

TaParser(list)
