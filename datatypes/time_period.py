from enum import Enum


class TimePeriod(Enum):
    r"""
    `TimePeriod` Enum that represents the time periods for the financial data.

    ## Options

    - **_1m** : `int` - 1 minute time period, 60 seconds.
    - **_5m** : `int` - 5 minutes time period, 300 seconds.
    - **_15m** : `int` - 15 minutes time period, 900 seconds.
    - **_30m** : `int` - 30 minutes time period, 1800 seconds.
    - **_1h** : `int` - 1 hour time period, 3600 seconds.
    - **_4h** : `int` - 4 hours time period, 14400 seconds.
    - **_12h** : `int` - 12 hours time period, 43200 seconds.
    - **_24h** : `int` - 24 hours time period, 86400 seconds.
    - **_48h** : `int` - 48 hours time period, 172800 seconds.
    - **_96h** : `int` - 96 hours time period, 345600 seconds.
    - **_1w** : `int` - 1 week time period, 604800 seconds.
    """

    _1m = 60
    _5m = 300
    _15m = 900
    _30m = 1800
    _1h = 3600
    _4h = 14400
    _12h = 43200
    _24h = 86400
    _48h = 172800
    _96h = 345600
    _1w = 604800
