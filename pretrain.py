import os
import random
from datetime import datetime

import lightning.pytorch as pl
import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from glm.model import Model
from glm.config import Config


def main():
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

    # pretrained = './albert_chinese_tiny'
    # train_df, valid_df, test_df = train.split_dataset(dataset_df)
    config = Config()
    model = Model(config)
    output_dir = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoint'),
        verbose=True,
        save_top_k=3,
        save_last=True,
        monitor="valid_mean_loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(monitor="valid_mean_loss", mode="min", patience=2, verbose=False)

    metrics_callback = MetricsCallback()

    trainer = pl.Trainer(
        max_epochs=8,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        default_root_dir=output_dir,
        accelerator='auto',
        num_sanity_val_steps=-1,
        callbacks=[metrics_callback, checkpoint_callback, early_stopping_callback]
    )
    trainer.fit(model=model)
    trainer.test(ckpt_path='best')

    metrics_callback.plot('mean_loss')

    best_state_dict = torch.load(checkpoint_callback.best_model_path)['state_dict']
    model.load_state_dict(best_state_dict)

    model.model.save_pretrained(os.path.join(output_dir, 'saved_model'))
    model.tokenizer.save_pretrained(os.path.join(output_dir, 'saved_model'))