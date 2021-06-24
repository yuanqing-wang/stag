import torch
from .layers import StagLayer
from .likelihoods import Likelihood, CategoricalLikelihood
from typing import Union, Callable, List

class StagModel(torch.nn.Module):
    def __init__(
        self,
        layers: List[StagLayer],
        likelihood: Likelihood=CategoricalLikelihood(),
        pool: Union[None, Callable]=None,
    ):
        super(StagModel, self).__init__()
        self.layers = layers
        self.likelihood = likelihood
        self.pool = pool

    def _forward(self, graph, feat):
        _graph = graph.local_var()
        for layer in self.layers:
            feat = layer(graph, feat)
        if self.pool is not None:
            g.ndata['h'] = feat
            feat = self.pool(g, 'h')
        return feat

    def forward(self, graph, feat, n_samples=1):
        feat = torch.mean(
            torch.stack(
                [
                    self._forward(graph, feat)
                    for _ in range(n_samples)
                ],
                dim=0,
            ),
            dim=0,
        )

        y_hat = self.likelihood.condition(feat).sample()
        return y_hat

    def loss(self, graph, feat, y, mask=None, n_samples=1):
        loss = 0.0
        for _ in range(n_samples):
            _feat = self._forward(graph, feat)
            nll = -self.likelihood.log_prob(_feat, y)
            if mask is not None:
                nll = nll[mask]
            nll = nll.sum()
            reg = 0.0
            for layer in self.layers:
                if hasattr(layer, "kl_divergence"):
                    reg += layer.kl_divergence()
            loss += nll + reg
        return loss / n_samples
