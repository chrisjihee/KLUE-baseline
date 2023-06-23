import logging
import sys
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    SKIP_KEYS = {"log", "progress_bar"}

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: int = 0) -> None:  # The `Callback.on_batch_end` hook was deprecated in v1.6 and will be removed in v1.8. Please use `Callback.on_train_batch_end` instead.
        # LR Scheduler
        lr_scheduler = trainer.lr_scheduler_configs[0].scheduler
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_last_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        global_step = pl_module.global_step
        if global_step == 0:
            return

        print(file=sys.stdout)
        print(file=sys.stderr)
        logger.info("***** Validation results *****")
        logger.info(f"global_step = {global_step}")
        metrics = trainer.callback_metrics
        for k, v in metrics.items():
            if k in self.SKIP_KEYS:
                continue
            logger.info(f"{k} = {v}")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        global_step = pl_module.global_step
        if global_step == 0:
            return

        print(file=sys.stdout)
        print(file=sys.stderr)
        logger.info("***** Test results *****")
        logger.info(f"global_step = {global_step}")
        metrics = trainer.callback_metrics
        for k, v in metrics.items():
            if k in self.SKIP_KEYS:
                continue
            logger.info(f"{k} = {v}")
