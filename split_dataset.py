import os
import re
from typing import List, Tuple
import random


def __split__and_write(
    data: List[str],
    path: str,
    dataset_name: str,
    ratio: Tuple[float, float, float],
    seed: int = 42,
):
    random.seed(seed)

    shuffled_data = data[:]
    random.shuffle(shuffled_data)

    total_size = len(shuffled_data)
    train_size = int(total_size * ratio[0])
    valid_size = int(total_size * ratio[1])

    train_data = shuffled_data[:train_size]
    valid_data = shuffled_data[train_size : train_size + valid_size]
    test_data = shuffled_data[train_size + valid_size :]

    valid_output_file_path = os.path.join(path, "validate", f"{dataset_name}.pretrain")
    test_output_file_path = os.path.join(path, "test", f"{dataset_name}.pretrain")
    train_output_file_path = os.path.join(path, "train", f"{dataset_name}.pretrain")

    try:
        with open(train_output_file_path, "w", encoding="utf-8") as train_file:
            train_file.write("\n".join(train_data))
        print(f"Training data written to {train_output_file_path}")

        with open(valid_output_file_path, "w", encoding="utf-8") as valid_file:
            valid_file.write("\n".join(valid_data))
        print(f"Validation data written to {valid_output_file_path}")

        with open(test_output_file_path, "w", encoding="utf-8") as test_file:
            test_file.write("\n".join(test_data))
        print(f"Test data written to {test_output_file_path}")

    except IOError as e:
        print(f"An error occurred while writing the files: {e}")


def __read_dataset_from_file(path: str) -> List[str]:
    dataset = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                if len(line) < 4096:
                    print(f"Skipping short line: {line}")
                    continue

                dataset.append(line)
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")

    return dataset


def construct_train_valid_test_datasets(path: str, seed: int = 42):
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "validate"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)

    pattern = r"^(.*)\.pretrain$"

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            print(f"Processing {file_name}")
            match = re.match(pattern, file_name)
            if match:
                name = match.group(1)
                print(f"Processing dataset: {name}...")

                dataset = __read_dataset_from_file(file_path)
                __split__and_write(dataset, path, name, (0.7, 0.15, 0.15), seed)
                dataset.clear()


if __name__ == "__main__":
    path = "/home/db/dev/TaSystem/data/pretrain/v1-small"
    construct_train_valid_test_datasets(path)
