import torch
from .util import get_data_parallel_group
from .sampler import RandomSampler, DistributedBatchSampler
from .blocklm_utils import ConstructBlockStrategy
from torch.utils.data import DataLoader
from .dataset import BlockDataset, PromptDataset
from .tokenizer import PretrainedChineseSPTokenizer as PretrainedCSP


def make_data_loader(dataset: BlockDataset, tokenizer: PretrainedCSP, batch_size: int,
                     args, shuffle: bool = False, block_collate: bool = False):
    world_size = torch.distributed.get_world_size(group=get_data_parallel_group())
    rank = torch.distributed.get_rank(group=get_data_parallel_group())
    if args.loader_scatter is not None:
        rank = rank // args.loader_scatter
        world_size = world_size // args.loader_scatter
        batch_size = batch_size // args.loader_scatter
    distributed = world_size > 1
    if shuffle:
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=batch_size * args.train_iters * args.gradient_accumulation_steps)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    drop_last = distributed
    # the GPUs in the same model parallel group receive the same data
    if distributed:
        batch_sampler = DistributedBatchSampler(sampler, batch_size, drop_last, rank,
                                                world_size,
                                                gradient_accumulation_steps=args.gradient_accumulation_steps)
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      batch_size,
                                                      drop_last)
    collate_fn = None
    if block_collate:
        collate_fn = ConstructBlockStrategy(args, tokenizer, args.seq_length, bert_prob=args.bert_prob,
                                            gap_sentence_prob=args.gap_sentence_prob,
                                            gap_sentence_ratio=args.gap_sentence_ratio,
                                            gpt_infill_prob=args.gpt_infill_prob,
                                            average_block_length=args.avg_block_length,
                                            gpt_min_ratio=args.gpt_min_ratio,
                                            block_mask_prob=args.block_mask_prob,
                                            context_mask_ratio=args.context_mask_ratio,
                                            short_seq_prob=args.short_seq_prob,
                                            single_span_prob=args.single_span_prob,
                                            shuffle_blocks=not args.no_shuffle_block,
                                            block_position_encoding=not args.no_block_position,
                                            sentinel_token=args.sentinel_token,
                                            encoder_decoder=args.encoder_decoder,
                                            task_mask=args.task_mask, random_position=args.random_position,
                                            masked_lm=args.masked_lm).construct_blocks
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             collate_fn=collate_fn)
    return data_loader

