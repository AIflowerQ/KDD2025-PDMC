import numpy as np
import torch
from ylearn.causal_discovery._discovery import DagNet

from causal_structure.api.causal_structure import CausalStructure


class DiscoveredCausalModel(CausalStructure):
    def __init__(self, scm: DagNet, dtype):
        super().__init__()

        self.scm: DagNet = scm
        self.device: torch.device = next(self.scm.parameters()).device

        self.dtype = dtype

    def causal_reconstruct(self, x: np.array) -> np.array:

        x_tensor: torch.Tensor = torch.tensor(data=x, dtype=self.dtype, device=self.device)

        rec_tensor: torch.Tensor = self.scm(x_tensor)

        rec_np: np.array = rec_tensor.detach().cpu().numpy()

        assert len(rec_np.shape) == 2

        return rec_np