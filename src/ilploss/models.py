import gurobi as grb
from gurobi import GRB
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import instantiate_class
from pytorch_lightning.utilities.model_summary import summarize
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm

import collections
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ILPModel(pl.LightningModule):
    def __init__(
        self,
        core: nn.Module,
        solver: nn.Module,
        optimizer: dict,
        lr_scheduler: dict,
        temp_scheduler: Optional[dict] = None,
    ):
        super().__init__()

        self.core = core
        self.solver = solver
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.temp_scheduler = temp_scheduler

        self.metrics = nn.ModuleDict(
            {
                "train/acc/idv": tm.Accuracy(
                    subset_accuracy=False,
                    mdmc_average="samplewise",
                ),
                "train/acc/all": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
                "val/acc/idv": tm.Accuracy(
                    subset_accuracy=False,
                    mdmc_average="samplewise",
                ),
                "val/acc/all": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "test/acc/idv": tm.Accuracy(
                    subset_accuracy=False,
                    mdmc_average="samplewise",
                ),
                "test/acc/all": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
            }
        )

    def configure_optimizers(self):
        optimizer = instantiate_class(self.parameters(), self.optimizer)

        config = self.lr_scheduler.pop("config", {})
        lr_scheduler = {
            "scheduler": instantiate_class(optimizer, self.lr_scheduler),
            **config,
        }

        if self.temp_scheduler is None:
            return [optimizer], [lr_scheduler]

        temp_init = self.temp_scheduler.pop("init")
        temp_optimizer = optim.SGD([nn.Parameter(torch.zeros(1))], lr=temp_init)
        temp_config = self.temp_scheduler.pop("config", {})
        temp_scheduler = {
            "scheduler": instantiate_class(temp_optimizer, self.temp_scheduler),
            **temp_config,
        }

        return [optimizer, temp_optimizer], [lr_scheduler, temp_scheduler]

    def temp(self):
        temp_scheduler = self.lr_schedulers()[1]
        return temp_scheduler.optimizer.param_groups[0]["lr"]

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        i, x, y = batch["idx"], batch["x"], batch["y"]
        a, b, c = self.core(i, x)
        yhat, status = self.solver(a, b, c, None)

        r = "train" if dataloader_idx == 0 else "val"
        lb = torch.minimum(torch.min(yhat).long(), batch["y"].long())
        self.metrics[f"{r}/acc/idv"](yhat.long() - lb, batch["y"].long() - lb)
        self.metrics[f"{r}/acc/all"](yhat.long() - lb, batch["y"].long() - lb)

    def validation_epoch_end(self, outputs):
        logger.info(f"Epoch {self.current_epoch}")
        for k, v in self.metrics.items():
            logger.info(f"{k}: {v.compute().item():.4f}")
        logger.info("")
        self.log_dict(self.metrics)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx != 0:
            return None
        i, x, y = batch["idx"], batch["x"], batch["y"]
        loss = self.core(i, x, y, temp=self.temp())
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        i, x, y = batch["idx"], batch["x"], batch["y"]
        a, b, c = self.core(i, x)
        yhat, status = self.solver(a, b, c, None)

        lb = torch.minimum(torch.min(yhat).long(), batch["y"].long())
        self.test_metrics["test/acc/idv"](yhat.long() - lb, batch["y"].long() - lb)
        self.test_metrics["test/acc/all"](yhat.long() - lb, batch["y"].long() - lb)

    def test_epoch_end(self, outputs):
        self.log_dict(self.test_metrics)
