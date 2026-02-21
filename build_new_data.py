from all import get
from typing import List
from datatypes import TOHLCVT, TimePeriod
from ta_finetune import TaParser


def run(sequences_dataset, tp):
    total = len(sequences_dataset)
    nan_count = 0

    parent_dir = "/home/db/TaSystem/data/finetune/realtest"

    with open(f"{parent_dir}/dataset_{tp}.finetune", "a") as file:
        for s, p, c in sequences_dataset:
            p = TimePeriod(p * 60)
            parser = TaParser(s, period=p, complete_df=c, finetune=True)
            result = parser()
            del parser
            if type(result) == str:
                if result == "":
                    print("Warning: Empty result!")
                    continue
                if "NaN" in result:
                    nan_count += 1
                    continue
                file.write(str(result) + "\n")
            else:
                raise ValueError("Invalid result!")

    print(f"Total NaN count: {nan_count} out of {total}")


def main(filename: str, timeperiod):
    groups, complete_df = get(
        filename, find_tp_with_regex=False, timeperiod=timeperiod, timestamp_len=13
    )
    print(len(groups))
    print(len(complete_df))
    sequences_dataset = []
    for g in groups:
        l = len(g)
        step = 10
        for sb in range(0, l, step):
            if sb + 160 < l:
                sequence: List[TOHLCVT] = g[sb : sb + 160]
                sequences_dataset.append((sequence, timeperiod, complete_df))

            if len(sequence) != 160:
                raise ValueError("Invalid length of group!")
    run(sequences_dataset, timeperiod)
    del groups, complete_df, sequences_dataset


if __name__ == "__main__":
    main("./btc_1.csv", 1)
    # main("./btc_5.csv", 5)
    # main("./btc_15.csv", 15)
    # main("./btc_30.csv", 30)
    # main("./btc_60.csv", 60)
    # main("./btc_240.csv", 240)
    # main("./btc_720.csv", 720)
    # main("./btc_1440.csv", 1440)
