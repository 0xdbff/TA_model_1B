from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from typing import List
import pandas as pd
import re
import os
import glob
import random


def _read_dataset_file(path: str) -> List[str]:
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


class TaDaset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        items: List[str] | None = None,
        file_path: str | None = None,
        streaming: bool = True,
    ):
        super().__init__()

        self.streaming = streaming

        if self.streaming:
            if file_path is None:
                raise ValueError("File path required for streaming dataset")
            self.file_path = file_path
            self.line_offsets = self._get_line_offsets()
            if self.line_offsets is None:
                raise ValueError("Line offsets not initialized")

        else:
            self.items = items
            self.line_offsets = None

        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length

    def _get_line_offsets(self):
        """Builds a list of file offsets for the start of each line."""
        offsets = []
        offset = 0
        if self.streaming and self.file_path is not None:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    offsets.append(offset)

                    offset += len(line.encode("utf-8"))
            return offsets

    def __len__(self):
        if self.streaming:
            if self.line_offsets is None:
                raise ValueError("Line offsets not initialized")
            return len(self.line_offsets)

        if self.items is None:
            raise ValueError("Items not initialized")
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        try:
            if self.streaming and self.line_offsets is not None:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    f.seek(self.line_offsets[idx])
                    item = f.readline().strip()
            elif self.items is not None:
                item = self.items[idx]
            else:
                raise ValueError("Items or file path not initialized")

            encoded = self.tokenizer(
                item,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
            }

        except Exception as e:
            raise ValueError(f"Error encoding item {idx}: {e}")

    @staticmethod
    def initialize_from_path(
        path: str,
        dataset_file_name: str,
        tokenizer: PreTrainedTokenizerFast,
        seed=42,
        n_chunks=10,
    ) -> 'TaDaset':
        if os.path.exists(os.path.join(path, dataset_file_name)):
            return TaDaset(
                file_path=os.path.join(path, dataset_file_name), tokenizer=tokenizer
            )
        N = n_chunks
        random.seed(seed)

        temp_file_names = [os.path.join(path, f"temp_{i}.txt") for i in range(N)]
        temp_file_handles = [open(fd, "w", encoding="utf-8") for fd in temp_file_names]

        input_files = glob.glob(os.path.join(path, "*.pretrain"))
        for infile in input_files:
            with open(infile, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    i = random.randint(0, N - 1)
                    temp_file_handles[i].write(line)

        for f in temp_file_handles:
            f.close()

        with open(
            os.path.join(path, dataset_file_name), "w", encoding="utf-8"
        ) as f_out:
            for i in range(N):
                temp_fname = f"temp_{i}.txt"
                temp_fname = os.path.join(path, temp_fname)

                with open(temp_fname, "r", encoding="utf-8") as f_in:
                    lines = f_in.readlines()

                random.seed(seed)
                random.shuffle(lines)

                f_out.writelines(lines)

                os.remove(temp_fname)

        return TaDaset(
            file_path=os.path.join(path, dataset_file_name), tokenizer=tokenizer
        )

    @staticmethod
    def initialize_from_path_dict(
        path: str, tokenizer: PreTrainedTokenizerFast, seed=42, shuffle=True
    ) -> dict[str, "TaDaset"]:
        """ """
        random.seed(seed)
        pattern = r"^(.*)\.pretrain$"
        complete_dataset: dict[str, "TaDaset"] = {}

        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                match = re.match(pattern, file_name)
                if match:
                    dataset = _read_dataset_file(file_path)
                    if shuffle:
                        random.shuffle(dataset)
                    complete_dataset[match.group(1)] = TaDaset(
                        items=dataset, tokenizer=tokenizer, streaming=False
                    )

        return complete_dataset


class TaFinetuneDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        items: List[str] | None = None,
        file_path: str | None = None,
        streaming: bool = True,
    ):
        super().__init__()

        self.streaming = streaming

        if self.streaming:
            if file_path is None:
                raise ValueError("File path required for streaming dataset")
            self.file_path = file_path
            self.line_offsets = self._get_line_offsets()
            if self.line_offsets is None:
                raise ValueError("Line offsets not initialized")

        else:
            self.items = items
            self.line_offsets = None

        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length

    def _get_line_offsets(self):
        """Builds a list of file offsets for the start of each line."""
        offsets = []
        offset = 0
        if self.streaming and self.file_path is not None:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    offsets.append(offset)

                    offset += len(line.encode("utf-8"))
            return offsets

    def __len__(self):
        if self.streaming:
            if self.line_offsets is None:
                raise ValueError("Line offsets not initialized")
            return len(self.line_offsets)

        if self.items is None:
            raise ValueError("Items not initialized")
        return len(self.items)

    def __getitem__(self, idx):
        try:
            if self.streaming and self.line_offsets is not None:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    f.seek(self.line_offsets[idx])
                    item = f.readline().strip()
            elif self.items is not None:
                item = self.items[idx]
            else:
                raise ValueError("Items or file path not initialized")

            encoded = self.tokenizer(
                item,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze()
            attention_mask = encoded["attention_mask"].squeeze()

            labels = input_ids.clone()
            # labels[:] = -100
            #
            attention_seq_len = attention_mask.sum().item()

            if attention_seq_len <= 1800:
                raise ValueError(f"Sequence too short: {attention_seq_len}")
            #     start_idx = attention_seq_len - 4
            #     labels[start_idx:attention_seq_len] = input_ids[
            #         start_idx:attention_seq_len
            #     ]
            # else:
            #     raise ValueError(f"Sequence too short: {attention_seq_len}")

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        except Exception as e:
            raise ValueError(f"Error encoding item {idx}: {e}")

    @staticmethod
    def initialize_from_path(
        path: str,
        dataset_file_name: str,
        tokenizer: PreTrainedTokenizerFast,
        seed=42,
        n_chunks=10,
    ):
        if os.path.exists(os.path.join(path, dataset_file_name)):
            return TaFinetuneDataset(
                file_path=os.path.join(path, dataset_file_name), tokenizer=tokenizer
            )
        N = n_chunks
        random.seed(seed)

        temp_file_names = [os.path.join(path, f"temp_{i}.txt") for i in range(N)]
        temp_file_handles = [open(fd, "w", encoding="utf-8") for fd in temp_file_names]

        input_files = glob.glob(os.path.join(path, "*.pretrain"))
        for infile in input_files:
            with open(infile, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    i = random.randint(0, N - 1)
                    temp_file_handles[i].write(line)

        for f in temp_file_handles:
            f.close()

        with open(
            os.path.join(path, dataset_file_name), "w", encoding="utf-8"
        ) as f_out:
            for i in range(N):
                temp_fname = f"temp_{i}.txt"
                temp_fname = os.path.join(path, temp_fname)

                with open(temp_fname, "r", encoding="utf-8") as f_in:
                    lines = f_in.readlines()

                random.seed(seed)
                random.shuffle(lines)

                f_out.writelines(lines)

                os.remove(temp_fname)

        return TaFinetuneDataset(
            file_path=os.path.join(path, dataset_file_name), tokenizer=tokenizer
        )

    @staticmethod
    def initialize_from_path_dummy(
        path: str,
        dataset_file_name: str,
        tokenizer: PreTrainedTokenizerFast,
        seed=42,
        n_chunks=1,
    ):
        # if os.path.exists(os.path.join(path, dataset_file_name)):
        #     return TaFinetuneDataset(
        #         file_path=os.path.join(path, dataset_file_name), tokenizer=tokenizer
        #     )
        # N = n_chunks
        # random.seed(seed)
        #
        # temp_file_names = [os.path.join(path, f"temp_{i}.txt") for i in range(N)]
        # temp_file_handles = [open(fd, "w", encoding="utf-8") for fd in temp_file_names]
        #
        # input_files = glob.glob(os.path.join(path, "*.pretrain"))
        # for infile in input_files:
        #     with open(infile, "r", encoding="utf-8") as f_in:
        #         c = 0
        #         for line in f_in:
        #             c += 1
        #
        #             if c == 20:
        #                 break
        #
        #             i = random.randint(0, N - 1)
        #             temp_file_handles[i].write(line)
        #
        # for f in temp_file_handles:
        #     f.close()
        #
        # with open(
        #     os.path.join(path, dataset_file_name), "w", encoding="utf-8"
        # ) as f_out:
        #     for i in range(N):
        #         temp_fname = f"temp_{i}.txt"
        #         temp_fname = os.path.join(path, temp_fname)
        #
        #         with open(temp_fname, "r", encoding="utf-8") as f_in:
        #             lines = f_in.readlines()
        #
        #         random.seed(seed)
        #         random.shuffle(lines)
        #
        #         f_out.writelines(lines)
        #
        #         os.remove(temp_fname)

        return TaFinetuneDataset(
            file_path=os.path.join(path, dataset_file_name), tokenizer=tokenizer
        )

    @staticmethod
    def initialize_from_path_dict(
        path: str, tokenizer: PreTrainedTokenizerFast, seed=42, shuffle=True
    ) -> dict[str, "TaFinetuneDataset"]:
        """ """
        random.seed(seed)
        pattern = r"^(.*)\.pretrain$"
        complete_dataset: dict[str, "TaFinetuneDataset"] = {}

        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                match = re.match(pattern, file_name)
                if match:
                    dataset = _read_dataset_file(file_path)
                    if shuffle:
                        random.shuffle(dataset)
                    complete_dataset[match.group(1)] = TaFinetuneDataset(
                        tokenizer=tokenizer, items=dataset, streaming=False
                    )

        return complete_dataset


# if __name__ == "__main__":
#     from datetime import datetime
#     from build_tokenizer import build_tokenizer
#
#     tokenizer = build_tokenizer()
#
#     train_dataset = TaFinetuneDataset.initialize_from_path(
#         "/home/db/dev/TaSystem/data/finetune/v1/train/",
#         tokenizer=tokenizer,
#         dataset_file_name="v1_train.finetune",
#         # seed=42,
#         n_chunks=10,
#     )
#
#     val_dataset = TaFinetuneDataset.initialize_from_path(
#         "/home/db/dev/TaSystem/data/finetune/v1/validate/",
#         tokenizer=tokenizer,
#         dataset_file_name="v1_validate.finetune",
#         seed=42,
#         n_chunks=10,
#     )
#
#     test_dataset = TaFinetuneDataset.initialize_from_path_dict(
#         "/home/db/dev/TaSystem/data/finetune/v1/test/",
#         tokenizer=tokenizer,
#         # seed=42,
#         shuffle=False,
#     )
#
#     import torch
#     from torch import return_types
#
#     # torch.set_printoptions(threshold=10000)
#
#     time = datetime.now()
#     data_20 = train_dataset[20]
#     print(f"Data at index 20:")
#     print(f"input_ids: {data_20['input_ids']}")
#     print(f"attention_mask: {data_20['attention_mask']}")
#     print(f"labels: {data_20['labels']}")
#     print(f"Time taken: {datetime.now() - time}\n")
#
#     time = datetime.now()
#     data_200 = train_dataset[200]
#     print(f"Data at index 200:")
#     print(f"input_ids: {data_200['input_ids']}")
#     print(f"attention_mask: {data_200['attention_mask']}")
#     print(f"labels: {data_200['labels']}")
#     print(f"Time taken: {datetime.now() - time}\n")
#
#     print(test_dataset)
#
#     for key, dataset in test_dataset.items():
#         print(f"Dataset: {key}")
#         time = datetime.now()
#         data_20 = dataset[20]
#         print(f"Data at index 20:")
#         print(f"input_ids: {data_20['input_ids']}")
#         print(f"attention_mask: {data_20['attention_mask']}")
#         print(f"labels: {data_20['labels']}")
#         print(f"Time taken: {datetime.now() - time}\n")
#
#         time = datetime.now()
#         data_200 = dataset[200]
#         print(f"Data at index 200:")
#         print(f"input_ids: {data_200['input_ids']}")
#         print(f"attention_mask: {data_200['attention_mask']}")
#         print(f"labels: {data_200['labels']}")
#         print(f"Time taken: {datetime.now() - time}\n")
