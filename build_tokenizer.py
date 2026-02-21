from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def build_tokenizer(
    vocab_size: int = 1024,
    model_max_length: int = 1860,
    pad_token: str = "<pad>",
    eos_token: str = "</s>",
    bos_token: str = "<s>",
    cls_token: str = "<cls>",
    sep_token: str = "<sep>",
    unk_token: str = "<unk>",
    save_path: str = "/home/db/dev/TaSystem/models/ta-model-base-v1",
) -> PreTrainedTokenizerFast:
    tokens = __load_static_tokens_list(vocab_size)
    vocab = {token: idx for idx, token in enumerate(tokens)}
    initial_tk = Tokenizer(WordLevel(vocab, unk_token=unk_token))

    initial_tk.pre_tokenizer = Whitespace()
    initial_tk.save(save_path + "/tokenizer.json")

    tokenizer = __load_fast_tokenizer(save_path + "/tokenizer.json")
    tokenizer.add_special_tokens(
        {
            "pad_token": pad_token,
            "unk_token": unk_token,
            "cls_token": cls_token,
            "sep_token": sep_token,
            "eos_token": eos_token,
            "bos_token": bos_token,
        }
    )
    tokenizer.model_max_length = model_max_length
    tokenizer.clean_up_tokenization_spaces = True
    tokenizer.create_token_type_ids_from_sequences = False
    tokenizer.save_pretrained(save_path)

    return tokenizer


def __load_fast_tokenizer(path: str):
    return PreTrainedTokenizerFast(
            tokenizer_file=path,
            clean_up_tokenization_spaces=False,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            add_bos_token=True,
            add_eos_token=False,
            use_default_system_prompt=False,
            legacy=None,
            add_prefix_space=None,
    )


def __load_static_tokens_list(vocab_size: int = 1024):
    tokens = [
        # -- Special tokens --
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<cls>",
        "<sep>",
        # -- Sign tokens --
        "+",
        "-",
        # -- Delimiter tokens --
        "{",
        "}",
        "|",
        "<<<",
        ">>>",
        # -- Trend tokens --
        "U",
        "D",
        "N",
        # -- Historical changes tokens --
        "LDC",
        "LWC",
        "LMC",
        "LQC",
        "L6MC",
        "LYC",
        "NaN",
        # -- Prediction tokens --
        "->",
    ]

    time_periods = [60, 300, 900, 1800, 3600, 14400, 43200, 86400]
    window_sizes = [5, 10, 20, 40, 80, 160]

    # -- 2 digit hexadecimal tokens --
    for i in range(256):
        tokens.append(f"{i:02x}")

    # -- Volume tokens --
    for i in range(0, 101):
        tokens.append(f"V_{i}")
        tokens.append(f"V_G5_{i}")

    # -- Trades tokens --
    for i in range(0, 101):
        tokens.append(f"T_{i}")
        tokens.append(f"T_G5_{i}")

    # -- RSI values tokens --
    for i in range(0, 101):
        tokens.append(f"r{i}")

    for ws in window_sizes:
        # -- Trend identifier (group) tokens --
        tokens.append(f"U_G{ws}")
        tokens.append(f"D_G{ws}")
        tokens.append(f"N_G{ws}")

        for tp in time_periods:
            # -- RSI identifier tokens --
            tokens.append(f"RSI{ws}_{tp}")
            # -- EMA identifier tokens --
            tokens.append(f"EMA{ws}_{tp}")
            # -- SMA identifier tokens --
            tokens.append(f"SMA{ws}_{tp}")

    additional_special_tokens = vocab_size - len(tokens)

    if additional_special_tokens < 0:
        raise ValueError("Vocabulary size is too small")

    for i in range(additional_special_tokens):
        tokens.append(f"<at_{i}>")

    if len(tokens) != vocab_size:
        raise ValueError("Invalid number of tokens")

    return tokens


def __tokenizer_test(tokenizer: PreTrainedTokenizerFast):
    file_path = "/home/db/dev/TaSystem/data/pretrain/v1-small/dataset_1.pretrain"

    total_sequences = 0
    total_tokens = 0
    total_unks = 0
    max_tokens = 0
    min_tokens = float("inf")
    skips = 0

    # Read the file and process each line
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                skips += 1
                continue

            if len(line) < 4096:
                skips += 1
                print(f"Skipping short line: {line}")
                continue

            encoding = tokenizer.encode(line)
            original = tokenizer.decode(encoding)

            if original != line:
                raise ValueError(f"Failed to encode line")

            if len(encoding) < 0:
                print(f"Failed to encode line: {line}")
                continue

            num_tokens = len(encoding)
            num_unks = encoding.count(tokenizer.unk_token_id)

            total_sequences += 1
            total_tokens += num_tokens
            total_unks += num_unks
            max_tokens = max(max_tokens, num_tokens)
            min_tokens = min(min_tokens, num_tokens)

    average_tokens = total_tokens / total_sequences if total_sequences > 0 else 0

    print(f"Total sequences processed: {total_sequences}")
    print(f"Total <UNK> tokens: {total_unks}")
    print(f"Average unks per sequence: {total_unks / total_sequences:.2f}")
    print(f"Average tokens per sequence: {average_tokens:.2f}")
    print(f"Maximum tokens in a sequence: {max_tokens}")
    print(f"Minimum tokens in a sequence: {min_tokens}")
    print(f"Skipped Sequences: {skips}")


if __name__ == "__main__":
    tokenizer = build_tokenizer()
    __tokenizer_test(tokenizer)
