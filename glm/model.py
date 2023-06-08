from multiprocessing import cpu_count
from typing import Any, Union, Dict, List, Tuple

import lightning.pytorch as pl
from .dataloader import get_dataloader
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModel, AutoTokenizer
from typing import Dict
from .config import Config


class Model(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(config.pretrained, trust_remote_code=True)
        self.config = config

        self.criterion = torch.nn.CrossEntropyLoss()

        self.cpu_count: int = cpu_count()

        self.cache: Dict[str, Any] = {}

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        input_ids, labels, loss_mask, attention_mask, position_ids = batch
        logits: torch.Tensor = self.forward(input_ids, position_ids, attention_mask)
        labels_one_hot: torch.Tensor = torch.nn.functional.one_hot(labels, logits.size(2))
        loss: torch.Tensor = self.criterion(logits, labels_one_hot.float())
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        outputs = self.training_step(batch, batch_idx)
        return outputs

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        outputs = self.training_step(batch, batch_idx)
        return outputs

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor, memory: List[torch.Tensor] = None) -> torch.Tensor:
        memory = memory or []
        model_output = self.model(input_ids, position_ids, attention_mask, memory)
        logits, memory = model_output.logits, model_output.mems
        return logits

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        return get_dataloader(self.tokenizer, self.config, 1, 0)
