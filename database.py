class TimeSeriesDatabase:

    initialized = False
    __raw_input_sequences = []

    async def get(self):

        return await self.__load_from_redis()

    async def get_all():
        pass

    async def __load():
        pass

    async def __load_all(self, load_cached=True, cache=False):
        pass

    async def __load_from_redis(self):
        pass

    async def __compute_standardized_values(self, data):
        # call TA lib
        pass


    def load_ohlcv_file(self, file_path):
        pass
