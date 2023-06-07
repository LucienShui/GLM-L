from multiprocessing import cpu_count
from typing import Any, Union, Dict, List, Tuple

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer, AlbertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from .dataset import PairTextDataset


class Model(pl.LightningModule):
    def __init__(self, pretrained: str, dataset: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                 max_length: int = 128, batch_size: int = 32, learning_rate: float = 1e-4):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = AlbertForSequenceClassification.from_pretrained(pretrained)

        self.train_df, self.valid_df, self.test_df = dataset

        self.criterion = torch.nn.CrossEntropyLoss()

        self.max_length: int = max_length
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size

        self.cpu_count: int = cpu_count()

        self.cache: Dict[str, Any] = {}

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        x, y = batch
        logits: torch.Tensor = self.forward(**x)
        loss: torch.Tensor = self.criterion(logits, y)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        outputs = self.training_step(batch, batch_idx)
        return outputs

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        outputs = self.training_step(batch, batch_idx)
        return outputs

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        huggingface_output: SequenceClassifierOutput = self.model(input_ids, attention_mask, token_type_ids)
        logits: torch.Tensor = huggingface_output.logits
        return logits

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            PairTextDataset(self.train_df, self.tokenizer, self.max_length), collate_fn=PairTextDataset.collate_batch,
            batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_count
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            PairTextDataset(self.valid_df, self.tokenizer, self.max_length), collate_fn=PairTextDataset.collate_batch,
            batch_size=self.batch_size, num_workers=self.cpu_count
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            PairTextDataset(self.test_df, self.tokenizer, self.max_length), collate_fn=PairTextDataset.collate_batch,
            batch_size=self.batch_size, num_workers=self.cpu_count
        )
