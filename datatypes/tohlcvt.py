from typing import NamedTuple


class TOHLCVT(NamedTuple):
    r"""
    `TOHLCVT` NamedTuple that represents a financial data record with Open,
    High, Low, Close, Volume, Trades indexed by Timestamps.

    ## Attributes

    - **Index** : `TimeStamp` - The index of the data record, timestamp identifying
        the the time.
    - **O** : `float` - The opening price of the asset.
    - **H** : `float` - The highest price of the asset during the time period.
    - **L** : `float` - The lowest price of the asset during the time period.
    - **C** : `float` - The closing price of the asset.
    - **V** : `float` - The trading volume of the asset during the time period.
    - **T** : `int` - The number of finished trades during the time period.
    """

    Index: int
    O: float
    H: float
    L: float
    C: float
    V: float
    T: int
    is_synthetic: bool = False
