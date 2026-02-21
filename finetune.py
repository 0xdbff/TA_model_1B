from transformers.trainer import Trainer, TrainingArguments
from transformers import (
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)

from model_config import TaModel
from dataset import TaFinetuneDataset
from build_tokenizer import build_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os


# class FinetuneTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")  # Shape: [batch_size, seq_len]
#         outputs = model(**inputs)
#         logits = outputs.get("logits")  # Shape: [batch_size, seq_len, vocab_size]
#
#         batch_size = labels.size(0)
#         seq_len = labels.size(1)
#
#         logits_4_list = []
#         labels_4_list = []
#         predicted_numbers = []
#         actual_numbers = []
#         missed_sign = False
#         max_distance_loss = 131070.0
#
#         distance_loss = torch.tensor(0.0, device=labels.device)
#
#         for i in range(batch_size):
#             labels_i = labels[i]  # Shape: [seq_len]
#             logits_i = logits[i]  # Shape: [seq_len, vocab_size]
#
#             valid_positions = (labels_i != -100).nonzero(as_tuple=False).squeeze(-1)
#
#             print(f"Number of valid positions {valid_positions}")
#
#             if valid_positions.numel() >= 4:
#                 last_four_positions = valid_positions[-4]
#
#                 logits_4 = logits_i[last_four_positions, :]  # Shape: [4, vocab_size]
#                 labels_4 = labels_i[last_four_positions]  # Shape: [4]
#
#                 print(len(logits_4[:valid_positions, :]))
#                 print(len(labels_4[:valid_positions]))
#
#                 print(len(logits))
#                 print(len(labels))
#
#                 print(f"last 4 logits -> {logits_4}")
#                 print(f"last 4 labels -> {labels_4}")
#
#                 logits_4_list.append(logits_4)
#                 labels_4_list.append(labels_4)
#
#                 sign_logits = logits_4[1, :]  # Second token is the sign
#                 predicted_sign_id = sign_logits.argmax(dim=-1).item()
#                 actual_sign_id = labels_4[1].item()
#
#                 best_logits = logits_4.argmax(dim=-1)  # Shape: [4]
#                 best_values = logits_4.max(dim=-1).values  # Shape: [4]
#
#                 print(f"Last 4 labels: {labels_4.tolist()}\n")
#
#                 for idx, (logit_row, best_value, best_index) in enumerate(
#                     zip(logits_4, best_values, best_logits)
#                 ):
#                     print(
#                         f"Position {last_four_positions[idx]}: "
#                         f"Best Match Logit: {best_value.item()}, Index: {best_index.item()}"
#                     )
#
#                 number_logits = logits_4[2:, :]  # Next two tokens are the numbers
#                 predicted_num_ids = number_logits.argmax(dim=-1).tolist()
#                 actual_num_ids = labels_4[2:].tolist()
#
#                 sample_missed_sign = False
#                 pred_number = None
#                 actual_number = None
#
#                 if predicted_sign_id == 6:
#                     pred_sign = "+"
#                 elif predicted_sign_id == 7:
#                     pred_sign = "-"
#                 else:
#                     pred_sign = None
#                     sample_missed_sign = True
#
#                     distance_loss += torch.tensor(
#                         max_distance_loss, device=labels.device
#                     )
#
#                 if actual_sign_id == 6:
#                     actual_sign = "+"
#                 elif actual_sign_id == 7:
#                     actual_sign = "-"
#                 else:
#                     actual_sign = None
#
#                 if any(id < 24 or id > 279 for id in predicted_num_ids):
#                     distance_loss += torch.tensor(
#                         max_distance_loss, device=labels.device
#                     )
#                     sample_missed_sign = True
#                 else:
#                     pred_num_values = [id - 24 for id in predicted_num_ids]
#                     actual_num_values = [id - 24 for id in actual_num_ids]
#
#                     if len(pred_num_values) != 2 or len(actual_num_values) != 2:
#                         print("This should not happen, invalid len for numbers")
#
#                     pred_number = pred_num_values[0] * 256 + pred_num_values[1]
#                     actual_number = actual_num_values[0] * 256 + actual_num_values[1]
#
#                     if pred_sign == "-":
#                         pred_number = -pred_number
#                     elif pred_sign != "+":
#                         distance_loss += torch.tensor(
#                             max_distance_loss, device=labels.device
#                         )
#                         sample_missed_sign = True
#
#                     if actual_sign == "-":
#                         actual_number = -actual_number
#                     # else actual_sign == '+', do nothing
#
#                     if pred_number is not None and not sample_missed_sign:
#                         predicted_numbers.append(pred_number)
#                         actual_numbers.append(actual_number)
#
#                     print(f"Pred number -> {pred_number}")
#                     print(f"Actual number number -> {actual_number}")
#
#                 # Update missed_sign if sample_missed_sign is True
#                 if sample_missed_sign:
#                     missed_sign = True
#
#             else:
#                 print("No valid 4 tokens, this should not happen!")
#                 # Handle cases where there are less than four valid tokens
#                 continue  # Skip this sample
#
#         if labels_4_list:
#             all_last_four_logits = torch.cat(
#                 logits_4_list, dim=0
#             )  # Shape: [N, vocab_size]
#             all_last_four_labels = torch.cat(labels_4_list, dim=0)  # Shape: [N]
#
#             loss_fct = nn.CrossEntropyLoss()
#             loss_ce = loss_fct(all_last_four_logits, all_last_four_labels)
#         else:
#             print("No valid labels, this should not happen!")
#             # No valid labels; set ce loss to 10
#             loss_ce = torch.tensor(20.0, device=labels.device)
#
#         # Compute distance-based loss (L1 loss)
#         if predicted_numbers:
#             predicted_numbers_tensor = torch.tensor(
#                 predicted_numbers, device=labels.device, dtype=torch.float
#             )
#             actual_numbers_tensor = torch.tensor(
#                 actual_numbers, device=labels.device, dtype=torch.float
#             )
#             l1_loss = F.l1_loss(predicted_numbers_tensor, actual_numbers_tensor)
#             distance_loss += l1_loss
#         else:
#             # All predictions invalid, set distance_loss to max_distance_loss
#             distance_loss += torch.tensor(
#                 max_distance_loss, device=labels.device, dtype=torch.float
#             )
#
#         # Compute total loss
#         total_loss = (0.8 * loss_ce) + (0.01 * distance_loss) * (
#             1.2 if missed_sign else 1
#         )
#
#         print(total_loss)
#
#         if return_outputs:
#             return total_loss, outputs
#         else:
#             return total_loss


class FinetuneTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")  # Shape: [batch_size, seq_len]
        input_ids = inputs.get("input_ids")  # Shape: [batch_size, seq_len]
        outputs = model(**inputs)
        logits = outputs.get("logits")  # Shape: [batch_size, seq_len, vocab_size]

        batch_size = labels.size(0)
        seq_len = labels.size(1)

        max_distance_loss = 131070.0
        total_loss = torch.tensor(0.0, device=labels.device)
        loss_fct = nn.CrossEntropyLoss()
        loss_fct_ce = nn.CrossEntropyLoss(ignore_index=-100)

        for i in range(batch_size):
            labels_i = labels[i]  # Shape: [seq_len]
            input_ids_i = input_ids[i]  # Shape: [seq_len]
            logits_i = logits[i]  # Shape: [seq_len, vocab_size]

            # Initialize per-sample variables
            distance_loss_i = torch.tensor(0.0, device=labels.device)
            sample_missed_sign = False
            predicted_numbers_i = []
            actual_numbers_i = []
            sample_skipped = False

            # Compute cross-entropy over all tokens for this sample
            labels_ce_i = input_ids_i[1:].clone()  # Shifted input_ids for labels
            logits_ce_i = logits_i[:-1, :]  # Shifted logits to match labels_ce_i

            # Ensure cross-entropy is only calculated on valid tokens
            # labels_ce_i[labels_i[1:] == -100] = -100

            # Reshape for CrossEntropyLoss
            logits_ce_flat_i = logits_ce_i.reshape(-1, logits_ce_i.size(-1))
            labels_ce_flat_i = labels_ce_i.reshape(-1)

            # Compute cross-entropy loss over all tokens for this sample
            total_cross_entropy_i = loss_fct_ce(logits_ce_flat_i, labels_ce_flat_i)

            valid_positions = (labels_i != -100).nonzero(as_tuple=False).squeeze(-1)

            if valid_positions.numel() >= 5:
                # Adjust positions to account for the shift
                last_four_positions = valid_positions[-5:-1]  # Positions t-1 to t-4
                logits_positions = last_four_positions  # Positions for logits (t-1)
                labels_positions = last_four_positions + 1  # Positions for labels (t)

                # Ensure indices are within bounds
                valid_indices = labels_positions < seq_len
                logits_positions = logits_positions[valid_indices]
                labels_positions = labels_positions[valid_indices]

                if logits_positions.numel() < 4:
                    sample_skipped = True
                else:
                    logits_4 = logits_i[logits_positions, :]  # Shape: [4, vocab_size]
                    labels_4 = labels_i[labels_positions]  # Shape: [4]

                    sign_logits = logits_4[1, :]  # Second token is the sign
                    predicted_sign_id = sign_logits.argmax(dim=-1).item()
                    actual_sign_id = labels_4[1].item()

                    number_logits = logits_4[2:, :]  # Next two tokens are the numbers
                    predicted_num_ids = number_logits.argmax(dim=-1).tolist()
                    actual_num_ids = labels_4[2:].tolist()

                    pred_number = None
                    actual_number = None

                    if predicted_sign_id == 6:
                        pred_sign = "+"
                    elif predicted_sign_id == 7:
                        pred_sign = "-"
                    else:
                        pred_sign = None
                        sample_missed_sign = True
                        distance_loss_i += torch.tensor(
                            max_distance_loss, device=labels.device
                        )

                    if actual_sign_id == 6:
                        actual_sign = "+"
                    elif actual_sign_id == 7:
                        actual_sign = "-"
                    else:
                        actual_sign = None

                    if any(id < 24 or id > 279 for id in predicted_num_ids):
                        distance_loss_i += torch.tensor(
                            max_distance_loss, device=labels.device
                        )
                        sample_missed_sign = True
                    else:
                        pred_num_values = [id - 24 for id in predicted_num_ids]
                        actual_num_values = [id - 24 for id in actual_num_ids]

                        pred_number = pred_num_values[0] * 256 + pred_num_values[1]
                        actual_number = (
                            actual_num_values[0] * 256 + actual_num_values[1]
                        )

                        if pred_sign == "-":
                            pred_number = -pred_number
                        elif pred_sign != "+":
                            distance_loss_i += torch.tensor(
                                max_distance_loss, device=labels.device
                            )
                            sample_missed_sign = True

                        if actual_sign == "-":
                            actual_number = -actual_number

                        if pred_number is not None and not sample_missed_sign:
                            predicted_numbers_i.append(pred_number)
                            actual_numbers_i.append(actual_number)
            else:
                sample_skipped = True

            # Compute per-sample loss
            if sample_skipped:
                loss_ce_predictions_i = torch.tensor(20.0, device=labels.device)
                distance_loss_i += torch.tensor(
                    max_distance_loss, device=labels.device, dtype=torch.float
                )
            else:
                loss_ce_predictions_i = loss_fct(logits_4, labels_4)

                # Compute distance-based loss (L1 loss)
                if predicted_numbers_i:
                    predicted_numbers_tensor = torch.tensor(
                        predicted_numbers_i, device=labels.device, dtype=torch.float
                    )
                    actual_numbers_tensor = torch.tensor(
                        actual_numbers_i, device=labels.device, dtype=torch.float
                    )
                    l1_loss_i = F.l1_loss(
                        predicted_numbers_tensor, actual_numbers_tensor
                    )
                    distance_loss_i += l1_loss_i
                else:
                    # Prediction invalid; set distance_loss_i to max_distance_loss
                    distance_loss_i += torch.tensor(
                        max_distance_loss, device=labels.device, dtype=torch.float
                    )

            # Compute per-sample total loss
            total_loss_i = (
                (0.6 * total_cross_entropy_i)
                + (0.9 * loss_ce_predictions_i)
                * (0.01 * distance_loss_i)
            )
            total_loss += total_loss_i

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    # def compute_metrics(self, eval_preds):
    #     logits, labels = eval_preds
    #     predictions = np.argmax(logits, axis=-1)
    #
    #     batch_size = labels.shape[0]
    #     predicted_numbers = []
    #     actual_numbers = []
    #     max_distance_loss = 131070.0  # Maximum possible distance
    #
    #     for i in range(batch_size):
    #         labels_i = labels[i]
    #         predictions_i = predictions[i]
    #
    #         # Find valid positions where labels are not -100
    #         valid_positions = (labels_i != -100).nonzero()[0]
    #
    #         # Check if there are at least four valid tokens
    #         if len(valid_positions) >= 4:
    #             last_four_positions = valid_positions[-4:]
    #
    #             labels_4 = labels_i[last_four_positions]
    #             predictions_4 = predictions_i[last_four_positions]
    #
    #             # Extract sign and number tokens
    #             predicted_sign_id = predictions_4[1]
    #             actual_sign_id = labels_4[1]
    #
    #             predicted_num_ids = predictions_4[2:]
    #             actual_num_ids = labels_4[2:]
    #
    #             # Process predicted sign
    #             if predicted_sign_id == 6:
    #                 pred_sign = "+"
    #             elif predicted_sign_id == 7:
    #                 pred_sign = "-"
    #             else:
    #                 pred_sign = None  # Invalid sign
    #
    #             # Process actual sign
    #             if actual_sign_id == 6:
    #                 actual_sign = "+"
    #             elif actual_sign_id == 7:
    #                 actual_sign = "-"
    #             else:
    #                 actual_sign = None  # Should not happen if labels are correct
    #
    #             # Map IDs to values: value = id - 24
    #             if any(id < 24 or id > 279 for id in predicted_num_ids):
    #                 # Invalid number IDs, set maximum loss
    #                 pred_number = None
    #             else:
    #                 pred_num_values = [id - 24 for id in predicted_num_ids]
    #                 actual_num_values = [id - 24 for id in actual_num_ids]
    #
    #                 # Combine the two numbers to form a single number (big-endian)
    #                 pred_number = pred_num_values[0] * 256 + pred_num_values[1]
    #                 actual_number = actual_num_values[0] * 256 + actual_num_values[1]
    #
    #                 # Apply sign
    #                 if pred_sign == "-":
    #                     pred_number = -pred_number
    #                 elif pred_sign != "+":
    #                     # Invalid sign, set maximum loss
    #                     pred_number = None
    #
    #                 if actual_sign == "-":
    #                     actual_number = -actual_number
    #                 # else actual_sign == '+', do nothing
    #
    #             if pred_number is not None:
    #                 predicted_numbers.append(pred_number)
    #                 actual_numbers.append(actual_number)
    #             else:
    #                 # Invalid prediction, set maximum loss
    #                 predicted_numbers.append(0)
    #                 actual_numbers.append(actual_number)
    #
    #         else:
    #             # Not enough valid tokens, skip
    #             continue
    #
    #     if len(predicted_numbers) > 0:
    #         predicted_numbers = np.array(predicted_numbers, dtype=float)
    #         actual_numbers = np.array(actual_numbers, dtype=float)
    #         mse = np.mean((predicted_numbers - actual_numbers) ** 2)
    #         mae = np.mean(np.abs(predicted_numbers - actual_numbers))
    #     else:
    #         mse = max_distance_loss
    #         mae = max_distance_loss
    #
    #     # Compute perplexity
    #     eval_loss = self.evaluate()["eval_loss"] if hasattr(self, "evaluate") else None
    #     perplexity = np.exp(eval_loss) if eval_loss is not None else None
    #     return {
    #         "mse": mse,
    #         "mae": mae,
    #         "perplexity": perplexity,
    #     }


class TaModelFinetune:
    _instance = None

    training_args: TrainingArguments
    model: PreTrainedModel
    trainer: Trainer
    tokenizer: PreTrainedTokenizerFast

    train_dataset: TaFinetuneDataset
    eval_dataset: dict["str", TaFinetuneDataset] | None = None
    test_dataset: dict["str", TaFinetuneDataset] | None = None

    data_collator: DataCollatorWithPadding

    def __init__(
        self,
        dataset_path: str,
        model_save_path: str,
        checkpoint_save_path: str,
        base_model_path: str,
    ):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.checkpoint_save_path = checkpoint_save_path
        self.base_model_path = base_model_path

        self.tokenizer = build_tokenizer(save_path=self.model_save_path)

        self.__initialize_components()
        self.initialized = True

    def __initialize_train_dataset(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        path = os.path.join(self.dataset_path, "train")
        self.train_dataset = TaFinetuneDataset.initialize_from_path(
            path=path,
            tokenizer=self.tokenizer,
            dataset_file_name="v1_train.finetune",
        )

    def __initialize_eval_dataset(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        path = os.path.join(self.dataset_path, "validate")
        self.eval_dataset = TaFinetuneDataset.initialize_from_path_dict(
            tokenizer=self.tokenizer,
            path=path,
        )

    def __initialize_test_dataset(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        path = os.path.join(self.dataset_path, "test")
        self.test_dataset = TaFinetuneDataset.initialize_from_path_dict(
            tokenizer=self.tokenizer,
            path=path,
        )

    def __initialize_test_dataset_recent(
        self,
        path="/home/db/TaSystem/data/finetune/realtest",
    ):
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        # path = os.path.join(self.dataset_path, "test")
        self.test_dataset_recent = TaFinetuneDataset.initialize_from_path_dict(
            tokenizer=self.tokenizer,
            path=path,
        )

    def __initialize_datacolator(self):
        # self.data_collator = DataCollatorWithPadding(
        #     tokenizer=self.tokenizer, padding="longest"
        # )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def __initialize_model(self):
        self.training_args = TrainingArguments(
            output_dir=self.checkpoint_save_path,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=True,
            evaluation_strategy="steps",
            eval_steps=50000,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=256,
            eval_accumulation_steps=8,
            torch_empty_cache_steps=256,
            learning_rate=9.99999e-5,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            num_train_epochs=1,
            lr_scheduler_type="linear",
            lr_scheduler_kwargs={},  #!TODO
            weight_decay=0.015,
            warmup_steps=20,
            log_level="debug",
            logging_steps=500,
            save_strategy="steps",
            save_steps=2000,
            # fp16=True,
            tf32=True,
            # bf16_full_eval=True,
            # half_precision_backend="auto",
            optim="adamw_torch_fused",
            resume_from_checkpoint=True,
            gradient_checkpointing=True,
            torch_compile=True,
        )
        model = TaModel(
            tokenizer=self.tokenizer, load_from_checkpoint=self.base_model_path
        )
        self.model = model()
        print(f"Model type : {type(self.model)}")
        # if isinstance(self.model, LlamaForCausalLM):
        #     raise ValueError('A pretrained base is required')

    def __initialize_components(self):
        self.__initialize_train_dataset()
        self.__initialize_eval_dataset()
        # self.__initialize_test_dataset_recent()

        # print(len(self.test_dataset_recent))

        self.__initialize_datacolator()
        self.__initialize_model()

        self.trainer = FinetuneTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            # compute_metrics=self.__compute_metrics,
        )

        # self.trainer = Trainer(
        #     model=self.model,
        #     args=self.training_args,
        #     data_collator=self.data_collator,
        #     train_dataset=self.train_dataset,
        #     eval_dataset=self.eval_dataset,
        #     tokenizer=self.tokenizer,
        #     # compute_metrics=self.__compute_metrics,
        # )

    # def __compute_metrics(self, eval_preds):
    #     return self.trainer.compute_metrics(eval_preds)

    def train(self, resume: bool = False):
        try:
            if resume:
                self.trainer.train(resume_from_checkpoint=False)
            else:
                self.trainer.train()
        except Exception as e:
            print(f"Error training model: {e}")

    def eval(self):
        try:
            results = self.trainer.evaluate()
            print("Evaluation results:", results)
            return results
        except Exception as e:
            print(f"Error training model: {e}")


if __name__ == "__main__":
    instance = TaModelFinetune(
        "/home/db/TaSystem/data/finetune/v1",
        "/home/db/TaSystem/models/ta-model-finetuned-v1.2",
        "/data/checkpoints/v1.2-finetuned",
        # "/data/checkpoints/v1.2-finetuned/checkpoint-2000",
        "/data/checkpoints/v1.2-finetuned/checkpoint-1",
    )
    instance.train(resume=False)
    instance.trainer.save_model()
    # instance.eval()
