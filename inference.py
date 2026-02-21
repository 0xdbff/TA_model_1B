import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from build_tokenizer import build_tokenizer

possible_time_periods = (1, 5, 15, 30, 60, 240, 720, 1440)


def get_info_for_last_tp(pair: str = "BTCEUR", timeperiod=60):
    if timeperiod not in possible_time_periods:
        raise ValueError("Invalid time period.")
    pass


def main(input_text: str):
    model_path = "/data/checkpoints/v2/checkpoint-2000"
    tokenizer_path = "/home/db/TaSystem/models/ta-model-finetuned-v1.2"

    tokenizer = build_tokenizer(save_path=tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    model.eval()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(input_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)

    topk_token1 = 2
    prob_values_token1, indices_token1 = torch.topk(probabilities, topk_token1)
    prob_values_token1 = prob_values_token1.squeeze()
    indices_token1 = indices_token1.squeeze()

    results = []

    for i in range(topk_token1):
        token1_id = indices_token1[i].unsqueeze(0)
        prob_token1 = prob_values_token1[i]

        input_ids_token1 = torch.cat([input_ids, token1_id.unsqueeze(0)], dim=1)

        with torch.no_grad():
            outputs_token1 = model(input_ids_token1)
            logits_token2 = outputs_token1.logits[:, -1, :]
            probabilities_token2 = torch.softmax(logits_token2, dim=-1)

        topk_token2 = 128
        prob_values_token2, indices_token2 = torch.topk(
            probabilities_token2, topk_token2
        )
        prob_values_token2 = prob_values_token2.squeeze()
        indices_token2 = indices_token2.squeeze()

        for j in range(topk_token2):
            token2_id = indices_token2[j].unsqueeze(0)
            prob_token2 = prob_values_token2[j]

            input_ids_token2 = torch.cat(
                [input_ids_token1, token2_id.unsqueeze(0)], dim=1
            )

            with torch.no_grad():
                outputs_token2 = model(input_ids_token2)
                logits_token3 = outputs_token2.logits[:, -1, :]
                probabilities_token3 = torch.softmax(logits_token3, dim=-1).squeeze()

            prob_token3 = probabilities_token3  # Shape: (vocab_size,)
            combined_prob = prob_token1 * prob_token2 * prob_token3

            token1_text = tokenizer.decode(token1_id)
            token2_text = tokenizer.decode(token2_id)

            token3_ids = torch.arange(probabilities_token3.size(0)).to(device)
            token3_texts = tokenizer.batch_decode(token3_ids.unsqueeze(1))

            for k in range(len(token3_texts)):
                token3_text = token3_texts[k]
                prob_token3_k = prob_token3[k]
                combined_prob_k = combined_prob[k]

                results.append(
                    {
                        "tokens": [token1_text, token2_text, token3_text],
                        "probability": combined_prob_k.item(),
                    }
                )

    results.sort(key=lambda x: x["probability"], reverse=True)

    top_N = 10
    for res in results[:top_N]:
        print(f"Tokens: {res['tokens']}, Probability: {res['probability']}")


def main2(input_text: str):
    model_path = "/data/checkpoints/v1.3-finetuned/checkpoint-2000"
    tokenizer_path = "/home/db/TaSystem/models/ta-model-finetuned-v1.2"

    tokenizer = build_tokenizer(save_path=tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    model.eval()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    num_tokens_to_generate = 3

    for _ in range(num_tokens_to_generate):
        with torch.inference_mode():
            outputs = model(input_ids)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            probabilities = torch.softmax(next_token_logits, dim=-1)

        print(probabilities[0, 6])
        print(probabilities[0, 7])

        # Get the token with the highest probability
        next_token_id = torch.argmax(probabilities, dim=-1)

        # Append the token to input_ids
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

    # Decode the generated tokens
    generated_tokens = input_ids[0, -num_tokens_to_generate:]
    generated_text = tokenizer.decode(generated_tokens)

    print(f"Generated tokens: {generated_text}")


if __name__ == "__main__":
    main2(
            "{ + 01 c7 U V_56 T_58 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 + 00 00 N V_46 T_44 | EMA5_60 + 01 aa SMA5_60 + 01 a9 U_G5 RSI5_60 r100 } { + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 - 01 ec D V_63 T_58 - 00 b9 D V_48 T_63 | EMA5_60 - 01 9f SMA5_60 - 01 78 D_G5 RSI5_60 r0 V_G5_50 T_G5_48 | EMA10_60 + 01 59 SMA10_60 + 01 8d U_G10 RSI10_60 r43 } { + 00 00 N V_51 T_58 + 00 00 N V_46 T_63 - 01 81 D V_100 T_100 - 01 ad D V_47 T_82 - 01 19 D V_46 T_54 | EMA5_60 - 01 bd SMA5_60 - 01 9b D_G5 RSI5_60 r0 } { + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 - 01 66 D V_61 T_49 - 01 53 D V_49 T_49 - 01 81 D V_46 T_44 | EMA5_60 - 01 9c SMA5_60 - 01 79 D_G5 RSI5_60 r0 V_G5_50 T_G5_44 | EMA10_60 - 02 11 SMA10_60 - 01 fa D_G10 RSI10_60 r0 | EMA20_60 - 01 db SMA20_60 - 01 80 D_G20 RSI20_60 r19 } { + 01 66 U V_51 T_49 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 | EMA5_60 + 00 00 SMA5_60 + 00 00 N_G5 RSI5_60 r50 } { - 01 60 D V_48 T_73 + 01 35 U V_47 T_87 + 00 00 N V_48 T_82 + 01 2c U V_48 T_54 + 00 00 N V_46 T_40 | EMA5_60 + 01 4f SMA5_60 + 01 43 U_G5 RSI5_60 r100 V_G5_47 T_G5_67 | EMA10_60 - 00 a6 SMA10_60 - 00 bc D_G10 RSI10_60 r55 } { + 00 e4 U V_46 T_44 - 01 0e D V_46 T_44 + 00 00 N V_46 T_40 - 01 d4 D V_46 T_44 + 01 ca U V_49 T_49 | EMA5_60 - 01 5c SMA5_60 - 01 53 D_G5 RSI5_60 r45 } { + 00 00 N V_46 T_40 - 01 9d D V_46 T_44 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 | EMA5_60 - 01 83 SMA5_60 - 01 82 D_G5 RSI5_60 r0 V_G5_46 T_G5_41 | EMA10_60 - 01 91 SMA10_60 - 01 87 D_G10 RSI10_60 r34 | EMA20_60 - 01 43 SMA20_60 - 01 24 D_G20 RSI20_60 r40 | EMA40_60 - 02 43 SMA40_60 - 02 23 D_G40 RSI40_60 r31 } { - 00 b9 D V_47 T_44 - 01 0e D V_57 T_49 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 | EMA5_60 - 00 fc SMA5_60 - 00 fc D_G5 RSI5_60 r0 } { + 00 00 N V_46 T_40 - 01 a1 D V_48 T_49 + 01 7c U V_46 T_44 + 01 45 U V_46 T_44 + 01 45 U V_46 T_44 | EMA5_60 + 00 d0 SMA5_60 - 00 d1 D_G5 RSI5_60 r62 V_G5_46 T_G5_44 | EMA10_60 - 00 ff SMA10_60 - 01 17 D_G10 RSI10_60 r56 } { + 01 6c U V_46 T_49 + 00 00 N V_46 T_40 + 01 60 U V_46 T_44 - 00 d1 D V_52 T_58 + 01 3d U V_48 T_54 | EMA5_60 + 01 4e SMA5_60 + 01 38 U_G5 RSI5_60 r91 } { + 01 53 U V_46 T_49 - 01 19 D V_49 T_49 + 01 0e U V_46 T_54 + 00 b9 U V_48 T_54 + 01 19 U V_46 T_44 | EMA5_60 + 00 c1 SMA5_60 + 00 5d U_G5 RSI5_60 r68 V_G5_47 T_G5_50 | EMA10_60 + 01 a2 SMA10_60 + 01 91 U_G10 RSI10_60 r85 | EMA20_60 + 01 bb SMA20_60 + 01 88 U_G20 RSI20_60 r73 } { + 00 00 N V_46 T_40 - 01 77 D V_46 T_49 - 01 4c D V_46 T_49 + 00 00 N V_46 T_44 - 00 d1 D V_46 T_44 | EMA5_60 - 01 97 SMA5_60 - 01 8f D_G5 RSI5_60 r0 } { + 00 00 N V_46 T_44 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 - 01 6c D V_47 T_54 + 01 0e U V_46 T_44 | EMA5_60 - 01 1b SMA5_60 - 01 04 D_G5 RSI5_60 r27 V_G5_46 T_G5_44 | EMA10_60 - 01 b8 SMA10_60 - 01 b5 D_G10 RSI10_60 r11 } { + 00 00 N V_46 T_40 - 01 6c D V_46 T_49 + 00 00 N V_46 T_49 + 01 6c U V_46 T_44 + 00 00 N V_46 T_40 | EMA5_60 - 00 ef SMA5_60 - 01 14 D_G5 RSI5_60 r50 } { + 00 00 N V_46 T_40 + 01 53 U V_46 T_49 - 00 96 D V_46 T_44 + 00 96 U V_47 T_49 - 01 0e D V_47 T_44 | EMA5_60 + 01 28 SMA5_60 + 01 30 U_G5 RSI5_60 r67 V_G5_46 T_G5_45 | EMA10_60 + 00 f9 SMA10_60 + 00 a6 U_G10 RSI10_60 r56 | EMA20_60 - 01 c0 SMA20_60 - 01 cd D_G20 RSI20_60 r33 | EMA40_60 + 01 a5 SMA40_60 + 01 a6 U_G40 RSI40_60 r56 | EMA80_60 - 02 4b SMA80_60 - 02 4f D_G80 RSI80_60 r42 } { - 01 a9 D V_51 T_68 - 02 07 D V_100 T_100 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 | EMA5_60 - 01 e6 SMA5_60 - 01 e5 D_G5 RSI5_60 r0 } { + 01 fd U V_47 T_49 - 01 19 D V_46 T_44 - 00 d1 D V_49 T_58 - 01 99 D V_48 T_44 + 00 00 N V_48 T_44 | EMA5_60 - 01 87 SMA5_60 - 01 6e D_G5 RSI5_60 r0 V_G5_48 T_G5_48 | EMA10_60 - 01 a2 SMA10_60 - 01 bb D_G10 RSI10_60 r36 } { + 00 00 N V_46 T_40 + 01 a6 U V_48 T_44 + 01 53 U V_56 T_44 + 01 02 U V_51 T_49 - 01 7c D V_46 T_44 | EMA5_60 + 01 a7 SMA5_60 + 01 a9 U_G5 RSI5_60 r70 } { - 01 e6 D V_46 T_44 + 01 ae U V_51 T_49 + 00 e4 U V_46 T_44 - 01 72 D V_100 T_77 + 01 2c U V_46 T_44 | EMA5_60 + 01 7a SMA5_60 + 01 80 U_G5 RSI5_60 r70 V_G5_58 T_G5_52 | EMA10_60 + 01 31 SMA10_60 + 01 62 U_G10 RSI10_60 r52 | EMA20_60 - 01 90 SMA20_60 - 01 a6 D_G20 RSI20_60 r45 } { - 01 6c D V_46 T_49 - 01 0e D V_48 T_44 - 01 2c D V_49 T_68 - 00 f4 D V_61 T_49 + 00 f4 U V_47 T_54 | EMA5_60 - 01 4a SMA5_60 - 01 42 D_G5 RSI5_60 r19 } { + 01 a6 U V_47 T_44 - 01 3d D V_47 T_49 + 01 02 U V_49 T_54 + 00 d1 U V_50 T_100 - 01 0e D V_53 T_58 | EMA5_60 - 01 03 SMA5_60 - 01 04 D_G5 RSI5_60 r32 V_G5_49 T_G5_61 | EMA10_60 + 00 a4 SMA10_60 - 00 c4 D_G10 RSI10_60 r52 } { - 01 81 D V_50 T_68 + 00 00 N V_46 T_40 - 01 8b D V_47 T_44 + 00 00 N V_46 T_40 + 00 f4 U V_46 T_44 | EMA5_60 - 01 59 SMA5_60 - 01 4c D_G5 RSI5_60 r17 } { - 01 9d D V_46 T_44 + 00 00 N V_46 T_40 + 01 02 U V_87 T_58 - 00 d1 D V_46 T_73 - 01 0e D V_47 T_77 | EMA5_60 + 00 60 SMA5_60 + 00 97 U_G5 RSI5_60 r38 V_G5_54 T_G5_58 | EMA10_60 - 01 b3 SMA10_60 - 01 a6 D_G10 RSI10_60 r15 | EMA20_60 - 01 b4 SMA20_60 - 01 91 D_G20 RSI20_60 r30 | EMA40_60 - 02 09 SMA40_60 - 01 f8 D_G40 RSI40_60 r39 } { + 00 e4 U V_62 T_49 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 - 01 e1 D V_47 T_49 + 00 00 N V_46 T_40 | EMA5_60 - 01 93 SMA5_60 - 01 6d D_G5 RSI5_60 r0 } { + 00 00 N V_46 T_40 + 00 00 N V_46 T_49 + 00 00 N V_46 T_40 + 01 60 U V_57 T_58 - 01 02 D V_46 T_44 | EMA5_60 + 01 13 SMA5_60 + 00 fc U_G5 RSI5_60 r74 V_G5_48 T_G5_46 | EMA10_60 - 01 a9 SMA10_60 - 01 a5 D_G10 RSI10_60 r24 } { + 01 a6 U V_46 T_44 - 01 35 D V_46 T_49 + 00 f4 U V_47 T_58 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 | EMA5_60 - 00 f8 SMA5_60 - 00 ff D_G5 RSI5_60 r31 } { + 01 bd U V_47 T_49 + 01 95 U V_51 T_54 - 00 e4 D V_96 T_73 + 00 b9 U V_46 T_44 - 01 35 D V_47 T_77 | EMA5_60 + 01 5c SMA5_60 + 01 65 U_G5 RSI5_60 r66 V_G5_57 T_G5_59 | EMA10_60 + 01 a2 SMA10_60 + 01 7a U_G10 RSI10_60 r73 | EMA20_60 + 01 04 SMA20_60 - 01 15 D_G20 RSI20_60 r60 } { - 00 d1 D V_47 T_91 + 00 00 N V_46 T_54 + 00 00 N V_46 T_40 + 00 e4 U V_50 T_44 + 00 00 N V_46 T_40 | EMA5_60 + 00 bf SMA5_60 + 00 ad U_G5 RSI5_60 r100 } { + 00 96 U V_47 T_73 + 01 60 U V_47 T_91 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 | EMA5_60 + 01 4a SMA5_60 + 01 49 U_G5 RSI5_60 r100 V_G5_46 T_G5_57 | EMA10_60 + 01 45 SMA10_60 + 01 2d U_G10 RSI10_60 r100 } { - 01 9d D V_49 T_54 - 01 9d D V_46 T_44 + 01 b5 U V_46 T_44 + 01 a2 U V_72 T_54 + 00 00 N V_46 T_40 | EMA5_60 + 01 66 SMA5_60 + 01 2a U_G5 RSI5_60 r69 } { + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 - 01 0e D V_46 T_44 + 00 00 N V_46 T_40 + 00 00 N V_46 T_40 | EMA5_60 - 00 f3 SMA5_60 - 00 e7 D_G5 RSI5_60 r0 V_G5_46 T_G5_41 | EMA10_60 + 01 83 SMA10_60 + 01 76 U_G10 RSI10_60 r64 | EMA20_60 + 01 4b SMA20_60 + 01 3d U_G20 RSI20_60 r57 | EMA40_60 + 01 a8 SMA40_60 + 01 73 U_G40 RSI40_60 r58 | EMA80_60 - 02 24 SMA80_60 - 02 2f D_G80 RSI80_60 r46 | EMA160_60 - 02 bb SMA160_60 - 02 b0 D_G160 RSI160_60 r44 } <<<LDC - 01 60 LWC - 03 d9 LMC + 06 d8 LQC + 04 54 L6MC + 0a 5f LYC + 0a 76 >>> -> "
    )
# + 01 9d
