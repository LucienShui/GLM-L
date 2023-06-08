import random
from bisect import bisect_right
from itertools import accumulate

import numpy as np
from torch.utils.data import Dataset
from .tokenizer import PretrainedChineseSPTokenizer as PretrainedCSP
from transformers import PreTrainedTokenizer
from json import loads

print_rank_0 = print


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.is_lazy = False
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._X = None
        self._Y = None
        self._lens = None

    def get_text_len(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_text_len(sample_idx)

    def SetTokenizer(self, tokenizer):
        for ds in self.datasets:
            ds.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.datasets[0].GetTokenizer()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def lens(self):
        if self._lens is None:
            self._lens = []
            if self.is_lazy:
                for data in self.datasets:
                    self._lens.extend(data.lens)
            else:
                for data in self.datasets:
                    self._lens.extend([len(d['text']) if isinstance(d, dict) else len(d) for d in data])
        return self._lens

    @property
    def X(self):
        if self._X is None:
            self._X = []
            for data in self.datasets:
                self._X.extend(data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = []
            for data in self.datasets:
                self._Y.extend(list(data.Y))
            self._Y = np.array(self._Y)
        return self._Y


class PromptDataset(Dataset):
    def __init__(self, filename: str, tokenizer: PreTrainedTokenizer = None):
        self.tokenizer = tokenizer
        self.prompt_list = []
        self.text_list = []
        with open(filename) as file:
            for line in file:
                json: dict = loads(line)
                self.prompt_list.append(tokenizer.encode(json['prompt'], add_special_tokens=False))
                self.text_list.append(tokenizer.encode(json['text'], add_special_tokens=False))
        self.prompt_length_list = list(map(len, self.prompt_list))
        self.text_length_list = list(map(len, self.text_list))

    def get_text_len(self, idx):
        return self.prompt_length_list[idx] + self.text_length_list[idx]

    def __getitem__(self, index):
        prompt = self.prompt_list[index]
        text = self.text_list[index]
        return {"tokens": prompt + text, "loss_masks": [0] * len(prompt) + [1] * len(text)}

    def __len__(self):
        return len(self.prompt_list)


class BlockDataset(Dataset):
    def __init__(self, dataset: ConcatDataset, tokenizer: PreTrainedTokenizer,
                 max_seq_len=1024,
                 sample_across_doc=True,
                 non_sentence_start_prob=0.0, filter_english=False):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = dataset
        self.ds_len = len(self.ds)
        self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.sample_across_doc = sample_across_doc
        self.non_sentence_start_prob = non_sentence_start_prob
        self.filter_english = filter_english
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        if self.filter_english:
            import fasttext
            self.model = fasttext.load_model('/mnt/lid.176.bin')
            print_rank_0("Load language detection model")
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
        self.total_len = np.sum(lens)
        print_rank_0(f"Dataset document count {len(lens)}, token count {self.total_len}, "
                     f"non sentence start {self.non_sentence_start_prob}")
        self.weighting = list(accumulate(lens))

    def get_weighted_samples(self, np_rng):
        while True:
            idx = np_rng.randint(self.total_len)
            data_idx = bisect_right(self.weighting, idx)
            tokens, loss_mask = self.getidx(data_idx)
            if self.filter_english:
                text = self.tokenizer.decode(tokens[:1024])
                lang = self.model.predict(text.replace('\n', ''))[0][0]
                if lang == '__label__en':
                    break
            else:
                break
        return tokens, loss_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        tokens, loss_mask = self.get_weighted_samples(rng)
        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len + 1

        # randomly choose a position for start
        if tokens_to_strip > 0:
            move_count = 0
            strip_left_tokens = rng.randint(tokens_to_strip)
            if rng.random() > self.non_sentence_start_prob:
                if rng.random() < 0.5:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens > 0 and not self.contains_sentence_end(
                            tokens[strip_left_tokens - 1]):
                        strip_left_tokens -= 1
                        move_count += 1
                else:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens < len(
                            tokens) and not self.contains_sentence_end(tokens[strip_left_tokens - 1]):
                        strip_left_tokens += 1
                        move_count += 1
            tokens = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + tokens[strip_left_tokens:]
            loss_mask = [0] + loss_mask[strip_left_tokens:]
            if len(tokens) == 2 and tokens[1] == self.tokenizer.convert_tokens_to_ids('[PAD]'):
                tokens, loss_mask = [], []
            tokens, loss_mask = self.right_strip_seq(tokens, loss_mask, self.max_seq_len)
        else:
            tokens = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + tokens
            loss_mask = [0] + loss_mask
            # Sample multiple documents
            if self.sample_across_doc:
                while len(tokens) < self.max_seq_len:
                    new_tokens, new_loss_mask = self.get_weighted_samples(rng)
                    new_tokens = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + new_tokens
                    new_loss_mask = [0] + new_loss_mask
                    is_last = len(new_tokens) >= self.max_seq_len - len(tokens)
                    new_tokens, new_loss_mask = self.right_strip_seq(new_tokens, new_loss_mask,
                                                                     self.max_seq_len - len(tokens))
                    tokens += new_tokens
                    loss_mask += new_loss_mask
                    if is_last:
                        break
        return {'text': np.array(tokens), "loss_mask": np.array(loss_mask)}

    def right_strip_seq(self, tokens, loss_mask, seq_length):
        strip_right_tokens = len(tokens) - seq_length
        if strip_right_tokens > 0:
            while strip_right_tokens < len(tokens) - 1 and not self.contains_sentence_end(
                    tokens[-strip_right_tokens - 1]):
                strip_right_tokens += 1
            if len(tokens) - strip_right_tokens < seq_length // 2:
                strip_right_tokens = len(tokens) - seq_length
            tokens = tokens[:-strip_right_tokens]
            loss_mask = loss_mask[:-strip_right_tokens]
        return tokens, loss_mask

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.convert_tokens_to_ids('[PAD]')]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.convert_tokens_to_ids('[PAD]') if pad_id is None else pad_id] * (num_pad_tokens)
        return seq

    # TODO: rewrite this function for chinese
    def contains_sentence_end(self, tok):
        tok = self.tokenizer.convert_ids_to_tokens(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        if '\n' in tok:
            return True
        return False
