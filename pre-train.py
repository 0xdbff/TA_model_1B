from transformers.trainer import Trainer, TrainingArguments
from transformers import (
    LlamaModel,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)
from model_config import TaModel
from dataset import TaDaset
from build_tokenizer import build_tokenizer
import os


class TaModelPretrainer:
    _instance = None

    training_args: TrainingArguments
    model: LlamaModel
    trainer: Trainer
    tokenizer: PreTrainedTokenizerFast

    train_dataset: TaDaset
    eval_dataset: dict["str", TaDaset]
    test_dataset: dict["str", TaDaset] | None = None

    data_collator: DataCollatorForLanguageModeling

    def __init__(
        self, dataset_path: str, model_save_path: str, checkpoint_save_path: str
    ):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.checkpoint_save_path = checkpoint_save_path

        self.tokenizer = build_tokenizer(save_path=self.model_save_path)

        self.__initialize_components()
        self.initialized = True

    def __initialize_train_dataset(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        path = os.path.join(self.dataset_path, "train")
        self.train_dataset = TaDaset.initialize_from_path(
            path=path,
            tokenizer=self.tokenizer,
            dataset_file_name="v1_train.pretrain",
        )

    def __initialize_eval_dataset(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        path = os.path.join(self.dataset_path, "validate")
        self.eval_dataset = TaDaset.initialize_from_path_dict(
            path=path, tokenizer=self.tokenizer
        )

    def __initialize_test_dataset(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        path = os.path.join(self.dataset_path, "test")
        self.eval_dataset = TaDaset.initialize_from_path_dict(path, self.tokenizer)

    def __initialize_datacolator(self):
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def __initialize_model(self):
        self.training_args = TrainingArguments(
            output_dir=self.checkpoint_save_path,
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            # do_predict=True,
            evaluation_strategy="steps",
            eval_steps=10000,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=3,
            gradient_accumulation_steps=256,
            eval_accumulation_steps=16,
            torch_empty_cache_steps=16,
            learning_rate=7.999999999e-5,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            num_train_epochs=1,
            # lr_scheduler_type="linear",
            lr_scheduler_type="cosine",
            # lr_scheduler_kwargs={},  #!TODO
            weight_decay=0.015,
            warmup_steps=100,
            log_level="debug",
            logging_steps=50,
            save_strategy="steps",
            save_steps=1000,
            # fp16=True,
            tf32=True,
            # bf16_full_eval=True,
            # half_precision_backend="auto",
            optim="adamw_torch_fused",
            resume_from_checkpoint=True,
            gradient_checkpointing=True,
            torch_compile=True,
        )
        model = TaModel(tokenizer=self.tokenizer)
        self.model = model()

    def __initialize_components(self):

        self.__initialize_train_dataset()
        self.__initialize_eval_dataset()
        self.__initialize_datacolator()

        self.__initialize_model()

        print(len(self.train_dataset))

        # model = LlamaForCausalLM.from_pretrained(self.model_save_path)
        # tokenizer = LlamaTokenizerFast.from_pretrained(self.model_save_path, legacy = False)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )

    def train(self):
        try:
            # self.trainer.train(resume_from_checkpoint=True)
            self.trainer.train()
        except Exception as e:
            print(f"Error training model: {e}")


if __name__ == "__main__":
    instance = TaModelPretrainer(
        "/home/db/TaSystem/data/v2",
        "/home/db/TaSystem/models/ta-model-base-v2",
        "/data/checkpoints/v2",
    )
    instance.train()
