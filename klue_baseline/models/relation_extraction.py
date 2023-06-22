import argparse
from typing import Dict, List, Tuple

import torch

from .mode import Mode
from .sequence_classification import SCTransformer


class RETransformer(SCTransformer):

    mode: str = Mode.RelationExtraction

    def __init__(self, hparams: argparse.Namespace, metrics: dict = {}) -> None:
        super().__init__(hparams, metrics=metrics)
        self.label_list = hparams.label_list

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        labels = torch.cat([output["labels"] for output in outputs], dim=0)
        preds, probs = self._convert_outputs_to_preds(outputs)

        if write_predictions is True:
            self.predictions = preds

        self._set_metrics_device()

        micro_f1 = self.metrics["micro_f1"]
        micro_f1.reset()
        micro_f1.update(preds, labels, self.label_list)
        self.log(f"{data_type}-micro_f1", micro_f1.compute(), on_step=False, on_epoch=True, logger=True)

        auprc = self.metrics["auprc"]
        auprc.reset()
        auprc.update(probs, labels)
        self.log(f"{data_type}-auprc", auprc.compute(), on_step=False, on_epoch=True, logger=True)

    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits: (B, num_labels)
        logits = torch.cat([output["logits"] for output in outputs], dim=0)
        return torch.argmax(logits, dim=1), torch.softmax(logits, dim=1)
