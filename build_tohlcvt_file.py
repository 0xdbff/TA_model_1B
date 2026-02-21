import re
import os

from datatypes import TOHLCVT
from typing import List
from all import get


def build_tohlcvt_file(
    parent_tohlcvt_file_path: str, timeperiod: int, output_file_path: str
):
    pattern = r"_(\d+)\.csv$"
    match = re.search(pattern, parent_tohlcvt_file_path)

    if not match:
        raise ValueError("Invalid filename!")

    orginal_time_period = int(match.group(1))

    time_diff_scale = timeperiod // orginal_time_period

    _, df = get(parent_tohlcvt_file_path)

    iteration = 1
    sub_list = []

    new_tohlcvt_list: List[TOHLCVT] = []

    for row in df.itertuples():
        row = TOHLCVT(*row)
        sub_list.append(row)

        if iteration % time_diff_scale == 0:

            if len(sub_list) != time_diff_scale:
                raise ValueError(
                    f"Invalid Data! len = { len(sub_list)} time_diff = {time_diff_scale} iteration = {iteration}"
                )

            for i in range(1, time_diff_scale):
                if sub_list[i].Index - sub_list[i - 1].Index != (
                    orginal_time_period * 60
                ):
                    raise ValueError(f"Invalid Data!")

            record = TOHLCVT(
                sub_list[0].Index,
                sub_list[0].O,
                max([r.H for r in sub_list]),
                min([r.L for r in sub_list]),
                sub_list[-1].C,
                sum([r.V for r in sub_list]),
                sum([r.T for r in sub_list]),
            )

            if record.V == 0 and record.T == 0:
                sub_list.clear()
                iteration += 1
                continue

            new_tohlcvt_list.append(record)
            sub_list.clear()

        iteration += 1

    with open(output_file_path, "w") as file:
        for record in new_tohlcvt_list:
            file.write(
                f"{record.Index},{record.O},{record.H},{record.L},{record.C},{record.V},{record.T}\n"
            )


if __name__ == "__main__":
    directory_path = "/home/db/data"

    pattern = r"(\w+)_1.csv$"
    size_threshold = 50 * 1024 * 1024

    extracted_names = []

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if re.match(pattern, file_name):
                file_path = os.path.join(root, file_name)
                if os.path.getsize(file_path) > size_threshold:
                    match = re.match(pattern, file_name)
                    if match:
                        extracted_name = match.group(1)
                        extracted_names.append(extracted_name)

    pattern = r"(\w+)_15.csv$"
    for _, _, files in os.walk(directory_path):
        for file_name in files:
            if re.match(pattern, file_name):
                file_path = os.path.join(directory_path, file_name)
                match = re.match(pattern, file_name)
                if match:
                    name = match.group(1)
                    if name in extracted_names:
                        print("processing", file_path)
                        output_file_path = os.path.join(
                            directory_path, f"{name}_30.csv"
                        )
                        build_tohlcvt_file(file_path, 30, output_file_path)

                        file_size = os.path.getsize(file_path)
                        output_file_size = os.path.getsize(output_file_path)

                        ratio = output_file_size / file_size
                        print(f"rationale {ratio}")

                        if ratio < 0.4:
                            raise ValueError("Dectected Errors!")

    pattern = r"(\w+)_60.csv$"
    for _, _, files in os.walk(directory_path):
        for file_name in files:
            if re.match(pattern, file_name):
                file_path = os.path.join(directory_path, file_name)
                match = re.match(pattern, file_name)
                if match:
                    name = match.group(1)
                    if name in extracted_names:
                        print("processing", file_path)
                        output_file_path = os.path.join(
                            directory_path, f"{name}_240.csv"
                        )
                        build_tohlcvt_file(file_path, 240, output_file_path)

                        file_size = os.path.getsize(file_path)
                        output_file_size = os.path.getsize(output_file_path)

                        ratio = output_file_size / file_size
                        print(f"rationale {ratio}")

                        if ratio < 0.19:
                            raise ValueError("Dectected Errors!")
