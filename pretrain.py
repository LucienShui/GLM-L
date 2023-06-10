import os
from datetime import datetime

import lightning.pytorch as pl
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from glm.model import Model
from glm.config import Config


def main():
    seed_everything(10086)

    config = Config().from_dict({
        'pretrained': 'glm-large-chinese',
        'dataset_list': ['test.jsonl'],
        'batch_size': 1,
        'num_workers': 0,
        'block_lm': True,
        'max_sequence_length': 128,
        'max_position_embeddings': 128
    })
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

    # metrics_callback = MetricsCallback()

    trainer = pl.Trainer(
        max_epochs=8,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        default_root_dir=output_dir,
        accelerator='auto',
        num_sanity_val_steps=-1,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    trainer.fit(model=model)
    trainer.test(ckpt_path='best')

    # metrics_callback.plot('mean_loss')

    best_state_dict = torch.load(checkpoint_callback.best_model_path)['state_dict']
    model.load_state_dict(best_state_dict)

    model.model.save_pretrained(os.path.join(output_dir, 'saved_model'))
    model.tokenizer.save_pretrained(os.path.join(output_dir, 'saved_model'))


if __name__ == '__main__':
    main()
