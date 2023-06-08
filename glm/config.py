from typing import List


class Config:
    def __init__(self):
        self.eod_token = None
        self.sample_across_doc: bool = False
        self.non_sentence_start_prob: float = 0.0
        self.max_sequence_length: int = 2048
        self.pretrained: str = ''
        self.dataset_list: List[str] = []

        self.transformer_xl: bool = False
        self.pretrained_bert: bool = False
        self.encoder_decoder: bool = False
        self.attention_dropout: float = 0.1
        self.num_attention_heads: int = 16
        self.hidden_size: int = 1024
        self.intermediate_size = None
        self.num_layers: int = 24
        self.layernorm_epsilon: float = 1e-05
        self.hidden_dropout: float = 0.1
        self.output_dropout: float = 0.1
        self.max_position_embeddings: int = 512
        self.vocab_size: int = 30522
        self.deep_init: bool = False
        self.make_vocab_size_divisible_by: int = 128
        self.cpu_optimizer: bool = False
        self.cpu_torch_adam: bool = False
        self.fp16: bool = False
        self.fp32_embedding: bool = False
        self.fp32_layernorm: bool = False
        self.fp32_tokentypes: bool = False
        self.fp32_allreduce: bool = False
        self.hysteresis: int = 2
        self.loss_scale = None
        self.loss_scale_window: int = 1000
        self.min_scale: int = 1
        self.attention_scale: float = 1.0
        self.experiment_name: str = "glm"
        self.batch_size: int = 4
        self.gradient_accumulation_steps: int = 1
        self.weight_decay: float = 0.01
        self.checkpoint_activations: bool = False
        self.checkpoint_num_layers: int = 1
        self.deepspeed_activation_checkpointing: bool = False
        self.epochs = None
        self.clip_grad: float = 1.0
        self.train_iters: int = 0
        self.label_smoothing: float = 0.0
        self.log_interval: int = 100
        self.summary_dir: str = ""
        self.seed: int = 1234
        self.reset_position_ids: bool = False
        self.reset_attention_mask: bool = False
        self.lr_decay_iters = None
        self.lr_decay_style: str = "linear"
        self.lr_decay_ratio: float = 0.1
        self.lr: float = 0.0001
        self.warmup: float = 0.01
        self.switch_linear: bool = False
        self.save = None
        self.new_save_directory: bool = False
        self.save_epoch: int = 1
        self.save_interval: int = 5000
        self.no_save_optim: bool = False
        self.no_save_rng: bool = False
        self.from_pretrained = None
        self.load = None
        self.no_load_optim: bool = False
        self.no_load_rng: bool = False
        self.no_load_lr_scheduler: bool = False
        self.no_deepspeed_load: bool = False
        self.finetune: bool = False
        self.resume_dataloader: bool = False
        self.distributed_backend: str = "nccl"
        self.DDP_impl: str = "torch"
        self.local_rank = None
        self.block_lm: bool = False
        self.masked_lm: bool = False
        self.bert_prob: float = 0.5
        self.gpt_infill_prob: float = 0.5
        self.gpt_min_ratio: float = 0.5
        self.gap_sentence_prob: float = 0.0
        self.gap_sentence_ratio: float = 0.15
        self.avg_block_length: int = 3
        self.short_seq_prob: float = 0.0
        self.single_span_prob: float = 0.0
        self.task_mask: bool = False
        self.no_shuffle_block: bool = False
        self.no_block_position: bool = False
        self.sentinel_token: bool = False
        self.block_mask_prob: float = 0.0
        self.context_mask_ratio: float = 0.0
        self.random_position: bool = False
        self.eval_batch_size = None
        self.eval_iters: int = 100
        self.eval_interval: int = 1000
        self.eval_epoch: int = 1
        self.eval_seq_length = None
        self.eval_max_preds_per_seq = None
        self.overlapping_eval: int = 32
        self.temperature: float = 1.0
        self.top_p: float = 0.0
        self.top_k: int = 0
        self.out_seq_length: int = 256
        self.num_beams: int = 1
        self.length_penalty: float = 0.0
        self.no_repeat_ngram_size: int = 0
        self.min_tgt_length: int = 0
        self.select_topk: bool = False
        self.blank_maskratio: float = 0.1
        self.model_parallel_size: int = 1
        self.shuffle: bool = False
        self.filter_english: bool = False
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.data_dir = None
        self.input_data_sizes_file: str = "sizes.txt"
        self.delim: str = ","
        self.text_key: str = "sentence"
        self.eval_text_key = None
        self.split: str = "1000,1,1"
        self.no_lazy_loader: bool = False
        self.half_lazy_loader: bool = False
        self.loader_scatter = None
        self.loose_json: bool = False
        self.presplit_sentences: bool = False
        self.num_workers: int = 2
        self.tokenizer_model_type = None
        self.tokenizer_path: str = "tokenizer.model"
        self.tokenizer_type: str = "BertWordPieceTokenizer"
        self.fix_command_token: bool = False
        self.no_pre_tokenize: bool = False
        self.cache_dir = None
        self.use_tfrecords: bool = False
        self.seq_length: int = 512
        self.mem_length: int = 0
        self.max_preds_per_seq = None
        self.non_sentence_start: float = 0.0
        self.sample_one_document: bool = False
        self.load_splits = None
        self.save_splits = None
        self.save_test_data = None
        self.multi_task_data = None
        self.multi_task_ratio: float = 0.0
        self.multi_seq_length = None
        self.multi_batch_size = None
        self.task = None
        self.load_pretrained = None
        self.pool_token: str = "cls"
        self.cloze_eval: bool = False
        self.multi_token: bool = False
        self.segment_length: int = 0
        self.loss_func: str = "cross_entropy"
        self.block_lm_ratio: float = 0.0
        self.adapet: bool = False
        self.pattern_id: int = 0
        self.fast_decode: bool = False
        self.few_superglue: bool = False
        self.eval_valid: bool = False
        self.validation_metric = None
        self.unidirectional: bool = False
        self.src_seq_length = None
        self.tgt_seq_length = None
        self.adam_beta1: float = 0.9
        self.adam_beta2: float = 0.999
        self.adam_eps: float = 1e-08
        self.optimizer: str = "adam"
        self.wsc_negative: bool = False
        self.overwrite: bool = False
        self.no_validation: bool = False
        self.continuous_prompt: bool = False
        self.num_prompt_tokens: int = 0
        self.prompt_func: str = "lstm"
        self.freeze_transformer: bool = False
        self.tune_prefix_layers = None
        self.prefix_prompt: int = 0
        self.prompt_init: bool = False
        self.mask_pad_token: bool = False
        self.deepspeed: bool = False
        self.deepspeed_config = None
        self.deepscale: bool = False
        self.deepscale_config = None
        self.deepspeed_mpi: bool = False
        self.cuda: bool = False
        self.rank: int = 0
        self.world_size: int = 1
        self.dynamic_loss_scale: bool = True

    def from_dict(self, config: dict):
        """
        Override if exists
        Args:
            config: config in dict type

        Returns:
            object itself
        """
        for k, v in config.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
