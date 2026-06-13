"""Optional EMA of model weights, enabled with ``pii-train --ema``.

Maintains a shadow copy of the parameters and, at the end of training, copies the
smoothed weights into the model so the saved checkpoint uses them.
"""

from __future__ import annotations

from transformers import TrainerCallback


class EMACallback(TrainerCallback):
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict = {}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def on_step_end(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach().float(), alpha=1.0 - self.decay
                )

    def on_train_end(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.dtype))
