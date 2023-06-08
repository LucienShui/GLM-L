from typing import Any, Union, Dict

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metric: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        # trainer 的 num_sanity_val_steps 不为 0 时，会在 train 之前跑一次 valid，造成 valid 多一次 metric
        # 为了消除这种影响，记录一下实际训练了多少 epoch
        self.epoch_cnt: int = 0

    @classmethod
    def to_numpy(cls, x: torch.Tensor) -> np.ndarray:
        return x.cpu().detach().numpy()

    def mean_loss(self, key: str) -> np.ndarray:
        mean_loss = np.mean(np.stack(self.cache[key]))
        return mean_loss

    def merge_tensor(self, key: str) -> np.ndarray:
        result = np.concatenate(self.cache[key], axis=0)
        return result

    def on_batch_end(self, outputs: Union[torch.Tensor, Dict[str, Any]], batch: Any, dataset_type: str) -> None:
        x, y_torch = batch
        loss: np.ndarray = self.to_numpy(outputs['loss'])
        logits: np.ndarray = self.to_numpy(outputs['logits'])
        y: np.ndarray = self.to_numpy(y_torch)
        self.cache.setdefault(dataset_type + '_loss', []).append(loss)
        self.cache.setdefault(dataset_type + '_logits', []).append(logits)
        self.cache.setdefault(dataset_type + '_label', []).append(y)

    def on_epoch_end(self, pl_module: pl.LightningModule, dataset_type: str) -> None:
        mean_loss: float = self.mean_loss(dataset_type + '_loss').item()
        logits: np.ndarray = self.merge_tensor(dataset_type + '_logits')
        y: np.ndarray = self.merge_tensor(dataset_type + '_label')

        accuracy, auc, metrics_df, confusion_df = calc_metrics(logits, y, 2)

        # print metrics to stdout
        delimiter = '=' * 64
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            print('', delimiter, str(metrics_df), delimiter, str(confusion_df), delimiter,
                  f'epoch = {pl_module.current_epoch}',
                  f'{dataset_type}_mean_loss = {mean_loss}',
                  f'{dataset_type}_accuracy = {accuracy}',
                  f'{dataset_type}_auc = {auc}', delimiter, sep='\n')
        pl_module.log_dict({
            dataset_type + '_mean_loss': mean_loss,
            dataset_type + '_accuracy': accuracy,
            dataset_type + '_auc': auc
        }, prog_bar=True, logger=True, sync_dist=True)

        # 如果是 num_sanity_val_steps 阶段，则 epoch 为 0
        self.metric.setdefault(dataset_type, {}).setdefault('x', []).append(
            self.epoch_cnt - (pl_module.global_step == 0))
        self.metric.setdefault(dataset_type, {}).setdefault('mean_loss', []).append(mean_loss)
        self.metric.setdefault(dataset_type, {}).setdefault('accuracy', []).append(accuracy)
        self.metric.setdefault(dataset_type, {}).setdefault('auc', []).append(auc)

        for key in ['_loss', '_logits', '_label']:
            self.cache[dataset_type + key] = []

    def on_train_batch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Union[torch.Tensor, Dict[str, Any]],
            batch: Any, batch_idx: int
    ) -> None:
        loss: np.ndarray = self.to_numpy(outputs['loss'])
        pl_module.log('train_loss', loss.item(), prog_bar=True, logger=True, sync_dist=True)
        self.cache.setdefault('train_loss', []).append(loss)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        mean_loss: float = self.mean_loss('train_loss').item()
        pl_module.log('train_mean_loss', mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.cache['train_loss'] = []

        self.metric.setdefault('train', {}).setdefault('x', []).append(self.epoch_cnt)
        self.metric.setdefault('train', {}).setdefault('mean_loss', []).append(mean_loss)
        self.epoch_cnt = pl_module.current_epoch + 1  # current_epoch 从 0 开始

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Union[torch.Tensor, Dict[str, Any]],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_end(outputs, batch, 'valid')

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.on_epoch_end(pl_module, 'valid')

    def on_test_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Union[torch.Tensor, Dict[str, Any]],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_end(outputs, batch, 'test')

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.on_epoch_end(pl_module, 'test')

    def plot(self, field: str) -> None:
        import matplotlib.pyplot as plt
        plt.title(field)
        plt.xlabel('epoch')

        plt.xticks(range(len(set(self.metric['train']['x'] + self.metric['valid']['x']))))
        plt.plot(self.metric['train']['x'], self.metric['train'][field])
        plt.plot(self.metric['valid']['x'], self.metric['valid'][field])
        plt.legend(['train', 'valid'])

        plt.show()
