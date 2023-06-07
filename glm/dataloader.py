import torch
from .sampler import RandomSampler, DistributedBatchSampler
from .blocklm_utils import ConstructBlockStrategy
from torch.utils.data import DataLoader
from .dataset import BlockDataset, PromptDataset
from .tokenizer import PretrainedChineseSPTokenizer as PretrainedCSP
from .config import Config


def make_data_loader(dataset: BlockDataset, tokenizer: PretrainedCSP, config: Config, world_size: int, rank: int):
    if config.loader_scatter is not None and config.loader_scatter > 0:
        rank = rank // config.loader_scatter
        world_size = world_size // config.loader_scatter
        batch_size = config.batch_size // config.loader_scatter
    else:
        rank = 0
        world_size = 1
        batch_size = config.batch_size
    distributed = world_size > 1
    if config.shuffle:
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=batch_size * config.train_iters * config.gradient_accumulation_steps)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    drop_last = distributed
    # the GPUs in the same model parallel group receive the same data
    if distributed:
        batch_sampler = DistributedBatchSampler(sampler, batch_size, drop_last, rank,
                                                world_size,
                                                gradient_accumulation_steps=config.gradient_accumulation_steps)
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      batch_size,
                                                      drop_last)
    collate_fn = None
    if config.block_collate:
        construct_blocks_fn = ConstructBlockStrategy(tokenizer, config.seq_length, config.eod_token,
                                                     bert_prob=config.bert_prob,
                                                     gap_sentence_prob=config.gap_sentence_prob,
                                                     gap_sentence_ratio=config.gap_sentence_ratio,
                                                     gpt_infill_prob=config.gpt_infill_prob,
                                                     average_block_length=config.avg_block_length,
                                                     gpt_min_ratio=config.gpt_min_ratio,
                                                     block_mask_prob=config.block_mask_prob,
                                                     context_mask_ratio=config.context_mask_ratio,
                                                     short_seq_prob=config.short_seq_prob,
                                                     single_span_prob=config.single_span_prob,
                                                     shuffle_blocks=not config.no_shuffle_block,
                                                     block_position_encoding=not config.no_block_position,
                                                     sentinel_token=config.sentinel_token,
                                                     encoder_decoder=config.encoder_decoder,
                                                     task_mask=config.task_mask, random_position=config.random_position,
                                                     masked_lm=config.masked_lm).construct_blocks

        def collate_fn(samples):
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(construct_blocks_fn(samples), config)
            return tokens, labels, loss_mask, attention_mask, position_ids
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=config.num_workers,
                             pin_memory=True,
                             collate_fn=collate_fn)
    return data_loader


def get_batch(data, config: Config):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    # keys = ['text', 'loss_mask']
    # if config.transformer_xl or config.block_lm:
    #     keys += ['target', 'attention_mask']
    # if config.block_lm:
    #     keys += ['position_id']
    # datatype = torch.int64

    # Broadcast data.
    # data_b = mpu.broadcast_data(keys, data, datatype)
    data_b = data
    # Unpack.
    # if config.transformer_xl:
    #     tokens = data_b['text'].long()
    #     labels = data_b['target'].long()
    #     attention_mask = data_b['attention_mask'].float()
    #     loss_mask = data_b['loss_mask'].float()
    # elif config.block_lm:
    tokens = data_b['text'].long()
    labels = data_b['target'].long()
    attention_mask = data_b['attention_mask'].long()
    loss_mask = data_b['loss_mask'].float()
    position_ids = data_b['position_id'].long()
    # else:
    #     tokens_ = data_b['text'].long()
    #     loss_mask = data_b['loss_mask'].float()
    #     labels = tokens_[:, 1:].contiguous()
    #     loss_mask = loss_mask[:, 1:].contiguous()
    #     tokens = tokens_[:, :-1].contiguous()
    #     attention_mask = None

    # Get the masks and postition ids.
    # if not config.block_lm:
    #     attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
    #         tokens,
    #         config.eod_token,
    #         config.reset_position_ids,
    #         config.reset_attention_mask,
    #         loss_mask=loss_mask,
    #         attention_mask=attention_mask,
    #         mem_length=config.mem_length,
    #         set_loss_mask=not config.transformer_xl)
    # Convert
    # if config.fp16:
    #     attention_mask = attention_mask.half()
    return tokens, labels, loss_mask, attention_mask, position_ids


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               set_loss_mask=False,
                               mem_length=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if mem_length:
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = torch.tril(torch.triu(attention_mask, 1 - seq_length + mem_length), mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if set_loss_mask:
        loss_mask[data == eod_token] = 0.0
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_dataloader(tokenizer: PretrainedCSP, config: Config, rank: int,
                   world_size: int) -> DataLoader:
    prompt_dataset = PromptDataset(config.dataset_list[0], tokenizer)
    block_dataset: BlockDataset = BlockDataset(
        prompt_dataset, tokenizer, max_seq_len=config.max_sequence_length,
        sample_across_doc=config.sample_across_doc, non_sentence_start_prob=config.non_sentence_start_prob)
    return make_data_loader(block_dataset, tokenizer, config, world_size, rank)
