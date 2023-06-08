import unittest


class TestArguments(unittest.TestCase):
    def test_arguments(self):
        from glm.argument import get_args

        args = get_args()
        for k, v in args.__dict__.items():
            if v is None:
                print(f'self.{k} = {v}')
            elif isinstance(v, str):
                print(f'self.{k}: {type(v).__name__} = "{v}"')
            else:
                print(f'self.{k}: {type(v).__name__} = {v}')


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        from transformers import AutoTokenizer, PreTrainedTokenizer
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('./glm-large-chinese', trust_remote_code=True)
        encoded_ids = tokenizer.encode('你好', add_special_tokens=False)
        print(encoded_ids)


class TestDataloader(unittest.TestCase):
    def test_dataloader(self):
        from glm.dataloader import get_dataloader
        from glm.config import Config
        from transformers import AutoTokenizer
        import torch
        import numpy as np
        import random
        from lightning import seed_everything

        seed = 10086
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        seed_everything(seed)

        config = Config().from_dict({
            'pretrained': 'glm-large-chinese',
            'dataset_list': ['test.jsonl'],
            'batch_size': 1,
            'num_workers': 0,
            'block_lm': True,
            "transformer_xl": False,
            "pretrained_bert": False,
            "encoder_decoder": False,
            "attention_dropout": 0.1,
            "num_attention_heads": 16,
            "hidden_size": 1024,
            "intermediate_size": None,
            "num_layers": 24,
            "layernorm_epsilon": 1e-05,
            "hidden_dropout": 0.1,
            "output_dropout": 0.1,
            "max_position_embeddings": 2048,
            "vocab_size": 50048,
            "deep_init": False,
            "make_vocab_size_divisible_by": 128,
            "cpu_optimizer": False,
            "cpu_torch_adam": False,
            "fp16": True,
            "fp32_embedding": False,
            "fp32_layernorm": False,
            "fp32_tokentypes": False,
            "fp32_allreduce": False,
            "hysteresis": 2,
            "loss_scale": None,
            "loss_scale_window": 1000,
            "min_scale": 1,
            "attention_scale": 1.0,
            # "experiment_name": blocklm - test - chinese06 - 08 - 20 - 48,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "weight_decay": 0.1,
            "checkpoint_activations": True,
            "checkpoint_num_layers": 1,
            "deepspeed_activation_checkpointing": True,
            "epochs": None,
            "clip_grad": 1.0,
            "train_iters": 10,
            "label_smoothing": 0.0,
            "log_interval": 50,
            # "summary_dir":,
            "seed": 1234,
            "reset_position_ids": False,
            "reset_attention_mask": False,
            "lr_decay_iters": 120000,
            # "lr_decay_style": cosine,
            "lr_decay_ratio": 0.1,
            "lr": 0.0001,
            "warmup": 0.04,
            "switch_linear": False,
            # "save": / model / checkpoints / blocklm - test - chinese06 - 0
            # 8 - 20 - 48,
            "new_save_directory": False,
            "save_epoch": 1,
            "save_interval": 500,
            "no_save_optim": False,
            "no_save_rng": False,
            # "from_pretrained": / model / glm - large - chinese,
            "load": None,
            "no_load_optim": False,
            "no_load_rng": False,
            "no_load_lr_scheduler": False,
            "no_deepspeed_load": False,
            "finetune": False,
            "resume_dataloader": True,
            # "distributed_backend": nccl,
            # "DDP_impl": torch,
            "local_rank": 0,
            "masked_lm": False,
            "bert_prob": 0.5,
            "gpt_infill_prob": 0.5,
            "gpt_min_ratio": 0.25,
            "gap_sentence_prob": 0.3,
            "gap_sentence_ratio": 0.15,
            "avg_block_length": 3,
            "short_seq_prob": 0.02,
            "single_span_prob": 0.0,
            "task_mask": True,
            "no_shuffle_block": False,
            "no_block_position": False,
            "sentinel_token": False,
            "block_mask_prob": 0.1,
            "context_mask_ratio": 0.0,
            "random_position": False,
            "eval_batch_size": None,
            "eval_iters": 100,
            "eval_interval": 500,
            "eval_epoch": 1,
            "eval_seq_length": None,
            "eval_max_preds_per_seq": None,
            "overlapping_eval": 32,
            "temperature": 1.0,
            "top_p": 0.0,
            "top_k": 0,
            "out_seq_length": 256,
            "num_beams": 1,
            "length_penalty": 0.0,
            "no_repeat_ngram_size": 0,
            "min_tgt_length": 0,
            "select_topk": False,
            "blank_maskratio": 0.1,
            "model_parallel_size": 1,
            "shuffle": False,
            "filter_english": False,
            # "train_data": ['test'],
            "valid_data": None,
            "test_data": None,
            "data_dir": None,
            # "input_data_sizes_file": sizes.txt,
            "delim": ",",
            # "text_key": sentence,
            "eval_text_key": None,
            # "split": 949, 50, 1,
            "no_lazy_loader": True,
            "half_lazy_loader": False,
            "loader_scatter": 1,
            "loose_json": False,
            "presplit_sentences": False,
            # "num_workers": 2,
            "tokenizer_model_type": None,
            # "tokenizer_path": tokenizer.model,
            # "tokenizer_type": PretrainedChineseSPTokenizer,
            "fix_command_token": False,
            "no_pre_tokenize": False,
            "cache_dir": None,
            "use_tfrecords": False,
            "seq_length": 512,
            "mem_length": 0,
            "max_preds_per_seq": None,
            "non_sentence_start": 0.0,
            "sample_one_document": False,
            "load_splits": None,
            "save_splits": None,
            "save_test_data": None,
            "multi_task_data": None,
            "multi_task_ratio": 0.0,
            "multi_seq_length": None,
            "multi_batch_size": None,
            "task": None,
            "load_pretrained": None,
            "pool_token": "cls",
            "cloze_eval": False,
            "multi_token": False,
            "segment_length": 0,
            "loss_func": "cross_entropy",
            "block_lm_ratio": 0.0,
            "adapet": False,
            "pattern_id": 0,
            "fast_decode": False,
            "few_superglue": False,
            "eval_valid": False,
            "validation_metric": None,
            "unidirectional": False,
            "src_seq_length": None,
            "tgt_seq_length": None,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-08,
            # "optimizer": adam,
            "wsc_negative": False,
            "overwrite": False,
            "no_validation": False,
            "continuous_prompt": False,
            "num_prompt_tokens": 0,
            # "prompt_func": lstm,
            "freeze_transformer": False,
            "tune_prefix_layers": None,
            "prefix_prompt": 0,
            "prompt_init": False,
            "mask_pad_token": False,
            "deepspeed": True,
            "deepscale": False,
            "deepscale_config": None,
            "deepspeed_mpi": False,
            "cuda": True,
            "rank": 0,
            "world_size": 1,
            "dynamic_loss_scale": True,
            "eod_token": 50000,
            "persist_state": 0,
            "lazy": False,
            "transpose": False,
            "data_set_type": "Block",
            "samples_per_shard": 100,
            "do_train": 1,
            "do_valid": 1,
            "do_test": 1,
            "iteration": 0,
        })
        tokenizer = AutoTokenizer.from_pretrained('./glm-large-chinese', trust_remote_code=True)
        dataloader = get_dataloader(tokenizer, config, 1, 0)
        for batch in dataloader:
            input_ids, labels, loss_mask, attention_mask, position_ids = batch
            break
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
