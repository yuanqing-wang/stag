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

    def loss_terms(self, graph, feat, y, mask=None, n_samples=1, kl_scaling=None):
        if kl_scaling is None: kl_scaling = self.kl_scaling
        total_nll = 0.0
        total_reg = 0.0
        for _ in range(n_samples):
            _feat = self._forward(graph, feat)
            nll = -self.likelihood.log_prob(_feat, y)
            if mask is not None:
                nll = nll[mask]
            nll = nll.mean()
            reg = 0.0
            for layer in self.layers:
                if layer.vi:
                    reg = reg + layer.kl_divergence()
            total_nll = total_nll + nll
            total_reg = total_reg + reg

        total_nll = total_nll / n_samples
        total_reg = total_reg / n_samples
        total_reg = total_reg * kl_scaling

        return total_nll, total_reg


    def loss(self, graph, feat, y, mask=None, n_samples=1, kl_scaling=None):
        nll, reg = self.loss_terms(graph, feat, y, mask=mask, n_samples=n_samples, kl_scaling=kl_scaling)
        return nll + reg


class StagModelContrastive(StagModel):
    def nll_contrastive(self, graph, feat):
        nll = 0.0
        for layer in self.layers:
            if hasattr(layer, "nll_contrastive"):
                nll = nll + layer.nll_contrastive(graph, feat)
        return nll
