import torch
from .layers import StagLayer
from .likelihoods import Likelihood, CategoricalLikelihood
from typing import Union, Callable, List

class StagModel(torch.nn.Module):
    def __init__(
        self,
        layers: List[StagLayer],
        likelihood: Likelihood=CategoricalLikelihood(),
        kl_scaling=1.0
    ):
        super(StagModel, self).__init__()
        self.layers = layers
        self.likelihood = likelihood
        self.kl_scaling = kl_scaling

    def _forward(self, graph, feat):
        _graph = graph.local_var()
        for layer in self.layers:
            feat = layer(_graph, feat)
        return feat

    def forward(self, graph, feat, n_samples=1, return_parameters=False):
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

        if return_parameters is True:
            return feat

        y_hat = self.likelihood.condition(feat).sample()
        return y_hat

    def loss(self, graph, feat, y, mask=None, n_samples=1, kl_scaling=None):
        if kl_scaling is None: kl_scaling = self.kl_scaling
        loss = 0.0
        for _ in range(n_samples):
            _feat = self._forward(graph, feat)
            nll = -self.likelihood.log_prob(_feat, y)
            if mask is not None:
                nll = nll[mask]
            nll = nll.mean()
            reg = 0.0
            for layer in self.layers:
                if layer.vi:
                    reg += layer.kl_divergence()
            loss += nll + kl_scaling * reg
        return loss / n_samples
