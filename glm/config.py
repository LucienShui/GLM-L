from typing import List


class Config:
    def __init__(self):
        self.batch_size = None
        self.loader_scatter = None
        self.train_iters = None
        self.gradient_accumulation_steps = None
        self.seq_length = None
        self.eod_token = None
        self.bert_prob = None
        self.gap_sentence_prob = None
        self.gap_sentence_ratio = None
        self.gpt_infill_prob = None
        self.avg_block_length = None
        self.gpt_min_ratio = None
        self.block_mask_prob = None
        self.context_mask_ratio = None
        self.short_seq_prob = None
        self.single_span_prob = None
        self.no_shuffle_block = None
        self.no_block_position = None
        self.sentinel_token = None
        self.encoder_decoder = None
        self.task_mask = None
        self.random_position = None
        self.masked_lm = None
        self.num_workers = None
        self.shuffle = None
        self.block_collate = None
        self.sample_across_doc: bool = None
        self.non_sentence_start_prob: float = None
        self.max_sequence_length: int = 2048
        self.pretrained: str = ''
        self.dataset_list: List[str] = []
