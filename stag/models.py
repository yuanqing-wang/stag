import torch
from .layers import StagLayer
from .likelihoods import Likelihood, CategoricalLikelihood
from typing import Union, Callable, List


def nll_contrastive(q_a, graph, feat):
    graph = graph.local_var()
    graph.ndata['h'] = feat

    # negative edge distribution
    fake_src = torch.randint(high=graph.number_of_nodes(), size=[graph.number_of_edges()])
    fake_dst = torch.randint(high=graph.number_of_nodes(), size=[graph.number_of_edges()])
    h_fake = q_a.embedding_mlp(torch.cat([graph.ndata['h'][fake_src], graph.ndata['h'][fake_dst]], dim=-1))
    fake_new_parameters = {key: q_a.parameters_mlp[key](h_fake) for key in q_a.new_parameter_names}
    q_a_negative = q_a.base_distribution_class(
            **{
                key.replace("log_", ""): fake_new_parameters[key].exp() if "log_" in key else fake_new_parameters[key]
                for key in q_a.new_parameter_names
            }
    )

    nll = -q_a.log_prob(torch.tensor(1.0, device=feat.device)) - q_a_negative.log_prob(torch.tensor(0.0, device=feat.device))
    nll = nll.sum(dim=-1).mean()
    return nll

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
        graph = graph.local_var()
        for layer in self.layers:
            feat = layer(graph, feat)
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
    def _forward(self, graph, feat):
        graph = graph.local_var()
        for layer in self.layers:
            _feat = layer(graph, feat)
            if hasattr(layer, "q_a"):
                _nll_contrastive = nll_contrastive(layer.q_a, graph, feat)
            else:
                _nll_contrastive = 0.0
            feat = _feat
        return feat, _nll_contrastive

    def loss_terms(self, graph, feat, y, mask=None, n_samples=1, kl_scaling=None):
        if kl_scaling is None: kl_scaling = self.kl_scaling
        total_nll = 0.0
        total_reg = 0.0
        for _ in range(n_samples):
            reg = 0.0
            _feat, nll_contrastive = self._forward(graph, feat)
            reg = reg + nll_contrastive
            nll = -self.likelihood.log_prob(_feat, y)
            if mask is not None:
                nll = nll[mask]
            nll = nll.mean()
            for layer in self.layers:
                if layer.vi:
                    reg = reg + layer.kl_divergence()
            total_nll = total_nll + nll
            total_reg = total_reg + reg

        total_nll = total_nll / n_samples
        total_reg = total_reg / n_samples
        total_reg = total_reg * kl_scaling

        return total_nll, total_reg

    def forward(self, graph, feat, n_samples=1, return_parameters=False):
        feat = torch.mean(
            torch.stack(
                [
                    self._forward(graph, feat)[0]
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


