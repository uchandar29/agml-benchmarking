from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
	import torch

	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def seed_dataloader_worker(worker_id: int) -> None:
	import torch

	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)
