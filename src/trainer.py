#!/usr/bin/env python3

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=pl.LightningModule,
        subclass_mode_model=True,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_data=True,
        run=False,
        save_config_overwrite=True,
    )
    cli.trainer.validate(cli.model, cli.datamodule)
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(datamodule=cli.datamodule)
