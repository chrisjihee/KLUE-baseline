from typing import Any, Callable, Optional

import torch
from pytorch_lightning.utilities import rank_zero_warn


class BaseMetric(torch.nn.Module):
    """Base class for metrics."""

    def __init__(
        self,
        metric_fn: Callable,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.preds = []
        self.targets = []

        rank_zero_warn(
            "MetricBase will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.metric_fn = metric_fn
        self.device = device

    def reset(self) -> None:
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates state with predictions and targets.

        Args:
            preds: Predictions from model
            targets: Ground truth values
        """

        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> Any:
        """Computes metric value over state."""

        preds = self.preds
        targets = self.targets

        if type(preds[0]) == torch.Tensor:
            preds = torch.cat(preds, dim=0)
            preds = preds.cpu().numpy()
        if type(targets[0]) == torch.Tensor:
            targets = torch.cat(targets, dim=0)
            targets = targets.cpu().numpy()

        score = self.metric_fn(preds, targets)
        return score


class LabelRequiredMetric(BaseMetric):
    """Metrics requiring label information."""

    def __init__(
        self,
        metric_fn: Callable,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            metric_fn=metric_fn,
            device=device,
        )
        self.label_info = None

    def update(self, preds: torch.Tensor, targets: torch.Tensor, label_info: Optional[Any] = None) -> None:
        """Updates state with predictions and targets.

        Args:
            preds: Predictions from model
            targets: Ground truth values
            label_info: Additional label information to compute the metric
        """

        self.preds.append(preds)
        self.targets.append(targets)
        if self.label_info is None:
            self.label_info = label_info

    def compute(self) -> Any:
        """Computes metric value over state."""

        preds = self.preds
        targets = self.targets

        if type(preds[0]) == torch.Tensor:
            preds = torch.cat(preds, dim=0)
            preds = preds.cpu().numpy()
        if type(targets[0]) == torch.Tensor:
            targets = torch.cat(targets, dim=0)
            targets = targets.cpu().numpy()

        score = self.metric_fn(preds, targets, self.label_info)
        return score
