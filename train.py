import os

import numpy as np
from pytorch_lightning import callbacks
import torch

from unet_lightning import LitUNet as Unet
from dataset import AgVisionDataSet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader


def main():
    torch.cuda.empty_cache()
    model = Unet()

    os.makedirs('lightning_logs', exist_ok=True)
    try:
        log_dir = sorted(os.listdir('lightning_logs'))[-1]
    except IndexError:
        log_dir = os.path.join('lightning_logs', 'version_0')

    os.makedirs('checkpoints', exist_ok=True)

    training_dataset = AgVisionDataSet(select_dataset='train', transform=None)
    val_dataset = AgVisionDataSet(select_dataset='val', transform=None)

    training_dataloader = DataLoader(dataset=training_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, num_workers=8)
        
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        save_top_k=3,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=8,
        verbose=True,
        check_on_train_epoch_end=True,
    )
    trainer = Trainer(
        gpus=2,
        callbacks=[checkpoint_callback, stop_callback],
        max_epochs=40,
    )

    trainer.fit(model, train_dataloader=training_dataloader, val_dataloaders=val_dataloader)
    print(checkpoint_callback.best_model_path, ": ", checkpoint_callback.best_model_score)


if __name__ == '__main__':
    main()
