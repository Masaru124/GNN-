"""Crystal GNN package root."""

import logging

# torch.utils.flop_counter logs "triton not found; flop counting will not work
# for triton kernels" at import time (pulled in via mlflow.pytorch ->
# pytorch_lightning). We never use flop counting, so silence it before any
# submodule can trigger that import.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

__all__ = []
