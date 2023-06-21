import logging
import sys

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    SKIP_KEYS = {"log", "progress_bar"}

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # LR Scheduler
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
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
