from transformers import (
    LlamaConfig,
    PreTrainedTokenizerFast,
    LlamaForCausalLM,
    PreTrainedModel,
)


class TaModel:
    _instance = None
    model: PreTrainedModel | LlamaForCausalLM | None = None

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        initialize_config=True,
        load_from_checkpoint: str | None = None,
    ):
        if not hasattr(self, "initialized"):
            if load_from_checkpoint is not None:
                self.model = LlamaForCausalLM.from_pretrained(load_from_checkpoint)
                print(f"Loaded from checkpoint{load_from_checkpoint}")
            else:
                self.tokenizer = tokenizer
                if initialize_config:
                    self.initialize_config()
            self.initialized = True
            return
        print("Already Initialized")

    def __call__(self) -> LlamaForCausalLM | PreTrainedModel:
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model

    def initialize_config(
        self,
        hidden_size: int = 2000,
        intermediate_size: int = 6000,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 20,
        num_key_value_heads: int | None = 20,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1860,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache=False,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.001,
        mlp_bias=False,
    ):
        """ """
        if self.tokenizer is None:
            raise ValueError(
                "A valid tokenizer is required for the model initialization"
            )

        if self.model:
            return self.model

        self.tokenizer.model_max_length = max_position_embeddings

        vocab_size = self.tokenizer.vocab_size
        pad_token_id = self.tokenizer.pad_token_id
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        if bos_token_id is None or eos_token_id is None:
            raise ValueError("Begin and end of sequence tokens are required")

        head_dim = hidden_size // num_attention_heads

        model_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
        )
        try:
            self.model = LlamaForCausalLM(model_config)

        except Exception as e:
            print(e)
            raise ValueError("Cannot proceed without a valid model configuration")


if __name__ == "__main__":
    """Test configuration, and print the number of parameters in the model"""
    # model = TaModel(tokenizer).initialize_config()
    # print(model)
    #
    # total_params = sum(param.numel() for param in model.parameters())
    # print(f"Total number of parameters in the model: {total_params / 10 **9}B")
    #
    # trainable_params = sum(
    #     param.numel() for param in model.parameters() if param.requires_grad
    # )
    # print(
    #     f"Total number of trainable parameters in the model {trainable_params / 10 **9}B"
    # )
    pass
