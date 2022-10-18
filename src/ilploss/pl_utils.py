import pytorch_lightning as pl
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning.utilities.model_summary import summarize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.data.dataloader import default_collate
import torchmetrics as tm

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import logging
from typing import Any, Callable, Union


logger = logging.getLogger(__name__)


class DictDataset(Dataset):
    def __init__(
        self,
        data: Mapping[str, Sequence[Any]],
    ):
        super().__init__()
        self.data = data
        assert len(data) != 0

        self.len = None
        for key, seq in data.items():
            if self.len is None:
                self.len = len(seq)
            else:
                assert self.len == len(seq)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ret = {}
        for key, seq in self.data.items():
            ret[key] = seq[idx]
        return ret


class BaseData(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        splits: Mapping[str, Union[int, float]],
        num_validate_train: Union[int, float],
        batch_sizes: Mapping[str, int],
        num_workers: int = 0,
        collate_fn: Callable = default_collate,
    ):
        super().__init__()

        self.data_path = data_path
        self.splits = splits
        self.num_validate_train = num_validate_train
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        logger.info(f"load {self.data_path}...")
        pt = torch.load(self.data_path, map_location="cpu")
        logger.info(f"loaded.")

        full = {}
        keys = []
        for key in ["x", "y"]:
            full[key] = pt[key]
            keys.append(key)
            assert len(full[key]) == len(full[keys[0]])

        full_len = len(full[keys[0]])

        if isinstance(self.splits["train"], float):
            assert isinstance(self.splits["val"], float) and isinstance(
                self.splits["test"], float
            )
            for s in ["train", "val", "test"]:
                self.splits[s] = int(self.splits[s] * full_len)

        assert (
            isinstance(self.splits["train"], int)
            and isinstance(self.splits["val"], int)
            and isinstance(self.splits["test"], int)
        )

        assert (
            len(full[keys[0]])
            >= self.splits["train"] + self.splits["val"] + self.splits["test"]
        )

        if isinstance(self.num_validate_train, float):
            assert self.num_validate_train <= 1.0
            self.num_validate_train = int(
                self.num_validate_train * self.splits["train"]
            )

        data = defaultdict(dict)
        for s in ["train", "val", "test"]:
            data[s]["idx"] = torch.arange(self.splits[s])

        for k in keys:
            data["train"][k] = full[k][: self.splits["train"]]
            data["val"][k] = full[k][
                self.splits["train"] : self.splits["train"] + self.splits["val"]
            ]
            data["test"][k] = full[k][-self.splits["test"] :]

        self.datasets = {}
        for s in ["train", "val", "test"]:
            self.datasets[s] = DictDataset(data[s])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.batch_sizes["train"],
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.splits["train"] % self.batch_sizes["train"] != 0,
            pin_memory=not isinstance(self.trainer.accelerator, CPUAccelerator),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset=Subset(
                    self.datasets["train"],
                    torch.multinomial(
                        torch.ones(self.splits["train"]),
                        num_samples=self.num_validate_train,
                        replacement=False,
                    ),
                ),
                batch_size=self.batch_sizes[
                    "val" if "val" in self.batch_sizes else "test"
                ],
                num_workers=self.num_workers,
                pin_memory=not isinstance(self.trainer.accelerator, CPUAccelerator),
                collate_fn=self.collate_fn,
            ),
            DataLoader(
                dataset=self.datasets["val"],
                batch_size=self.batch_sizes[
                    "val" if "val" in self.batch_sizes else "test"
                ],
                num_workers=self.num_workers,
                pin_memory=not isinstance(self.trainer.accelerator, CPUAccelerator),
                collate_fn=self.collate_fn,
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.batch_sizes["test"],
            num_workers=self.num_workers,
            pin_memory=not isinstance(self.trainer.accelerator, CPUAccelerator),
            collate_fn=self.collate_fn,
        )


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        optimizer: dict,
        lr_scheduler: dict,
        schedulers: dict = {},
    ):
        super().__init__()

        torch.backends.cudnn.benchmark = True

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.schedulers = schedulers

        self.metrics = nn.ModuleDict({})
        self.test_metrics = nn.ModuleDict({})

        self.my_hparams = {
            "optimizer": hparams(optimizer),
            "lr_scheduler": hparams(lr_scheduler),
            "schedulers": hparams(schedulers),
        }
        self.hparam_metrics = []

    def training_step_end(self, loss):
        if loss is not None:
            self.log("train/loss", loss)
        g = [p.grad.view(-1) for p in self.parameters() if p.grad is not None]
        if g:
            self.log("train/grad", torch.linalg.vector_norm(torch.cat(g)))

    def validation_epoch_end(self, outputs=None):
        self.log_dict(self.metrics)

    def test_epoch_end(self, outputs=None):
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        custom = self.optimizer.pop("custom", {})

        all_names = set()
        reg_names = set()
        params = defaultdict(list)
        for name, param in self.named_parameters():
            if param.requires_grad:
                all_names.add(name)
                for regex in custom.keys():
                    if re.fullmatch(regex, name):
                        reg_names.add(name)
                        params[regex].append(param)

        for regex in custom.keys():
            if regex not in params.keys():
                logger.warning(f"regex '{regex}' does not match any param!")

        groups = [
            {"params": params[regex], **kwargs} for regex, kwargs in custom.items()
        ]

        def_names = all_names - reg_names
        def_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name in def_names:
                    def_params.append(param)

        groups.append({"params": def_params})

        optimizer = instantiate_class(groups, self.optimizer)

        config = self.lr_scheduler.pop("config", {})
        lr_scheduler = {
            "scheduler": instantiate_class(optimizer, self.lr_scheduler),
            **config,
        }

        schedulers = []
        self.scheduler_name_to_id = {}
        dummy_optimizers = []
        for i, (name, scheduler) in enumerate(self.schedulers.items()):
            self.scheduler_name_to_id[name] = i
            init = scheduler.pop("init", 1.0)
            dummy_optimizers.append(optim.SGD([nn.Parameter(torch.zeros(1))], lr=init))
            config = scheduler.pop("config", {})
            schedulers.append(
                {
                    "scheduler": instantiate_class(dummy_optimizers[-1], scheduler),
                    **config,
                },
            )

        return [optimizer, *dummy_optimizers], [lr_scheduler, *schedulers]

    def query_scheduler(self, name):
        scheduler = self.lr_schedulers()[self.scheduler_name_to_id[name] + 1]
        return scheduler.optimizer.param_groups[0]["lr"]
