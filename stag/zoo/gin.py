import torch
import dgl

class GIN(dgl.nn.GINConv):
    def __init__(
        self,
        in_features, out_features,
        *args, **kwargs,
    ):
        apply_func = torch.nn.Linear(in_features, out_features)
        super().__init__(apply_func=apply_func, *args, **kwargs)
