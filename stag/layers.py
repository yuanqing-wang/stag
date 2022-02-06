import math
import torch
import dgl
from typing import Union
from .distributions import Distribution, ParametrizedDistribution


def _in_norm(graph, edge_weight_sample):
    graph = graph.local_var()
    # put weight into edges to normalize
    graph.edata["h_a"] = edge_weight_sample
    graph.update_all(
        dgl.function.copy_edge("h_a", "m_a"),
        dgl.function.sum("m_a", "h_a"),
    )

    # (n_nodes, )
    current_sum = graph.ndata["h_a"]

    # (n_nodes, )
    desired_sum = graph.in_degrees().unsqueeze(-1)

    # (n_nodes, )
    node_scaling = torch.where(
        torch.ne(current_sum, 0.0),
        desired_sum / current_sum,
        torch.zeros_like(current_sum),
    )

    graph.ndata["s"] = node_scaling

    # put scaling back to edges
    graph.apply_edges(lambda edges: {"s": edges.dst["s"]})
    edge_scaling = graph.edata["s"]
    edge_weight_sample *= edge_scaling
    return edge_weight_sample

class StagLayer(torch.nn.Module):
    """ Make a DGL Graph Conv Layer stochastic.

    Parameters
    ----------
    base_layer : torch.nn.Module
        A DGL Graph Conv layer.

    q_a : torch.distributions.Distribution
        Edge weight distribution.

    Methods
    -------
    forward(graph, feat)
        Forward pass.

    """
    def __init__(
        self,
        base_layer: torch.nn.Module,
        q_a: Union[Distribution, torch.distributions.Distribution]=torch.distributions.Normal(1.0, 1.0),
        p_a: Union[None, Distribution, torch.distributions.Distribution]=torch.distributions.Normal(1.0, 1.0),
        norm: bool=False,
        relu: bool=False,
        vi: bool=False,
    ) -> None:
        super(StagLayer, self).__init__()
        self.base_layer = base_layer

        if isinstance(q_a, torch.distributions.Distribution):
            q_a = ParametrizedDistribution(q_a, vi=vi)
        if isinstance(p_a, torch.distributions.Distribution):
            p_a = ParametrizedDistribution(p_a, vi=vi)
        elif p_a is None:
            p_a = ParametrizedDistribution(q_a, vi=vi)

        self.q_a = q_a
        self.p_a = p_a
        self.norm = norm
        self.relu = relu
        self.vi = vi

    def forward(self, graph, feat):
        """ Forward pass. """
        graph = graph.local_var()
        # rsample noise
        edge_weight_sample = self.rsample_noise(graph, feat)

        if self.relu:
            edge_weight_sample = edge_weight_sample.relu()

        # normalize so that for each node the sum of in_degrees are the same
        if self.norm:
            edge_weight_sample = _in_norm(
                graph, edge_weight_sample,
            )

        self._edge_weight_sample = edge_weight_sample

        return self.base_layer.forward(
            graph=graph,
            feat=feat,
            edge_weight=edge_weight_sample,
        )

    def rsample_noise(self, graph, feat):
        batch_shape = self.q_a.batch_shape

        if batch_shape == torch.Size([]):
            edge_weight_sample = self._rsample_noise_r1(graph, feat)
        elif batch_shape == torch.Size([feat.shape[1]]):
            edge_weight_sample = self._rsample_noise_rc(graph, feat)
        elif batch_shape == torch.Size([graph.number_of_edges()]):
            edge_weight_sample = self._rsample_noise_re(graph, feat)
        elif batch_shape == torch.Size(
            [graph.number_of_edges(), feat.shape[1]]
        ):
            edge_weight_sample = self._rsample_noise_rec(graph, feat)

        return edge_weight_sample

    def _rsample_noise_r1(self, graph, feat):
        """ Sample from a distribution on $\mathbb{R}^1$. """
        return self.q_a.rsample(
            [graph.number_of_edges(), feat.shape[1]],
        )

    def _rsample_noise_rc(self, graph, feat):
        """ Sample from a distribution on $\mathbb{R}^C$. """
        return self.q_a.rsample(
            [graph.number_of_edges()],
        )

    def _rsample_noise_re(self, graph, feat):
        """ Sample from a distribution on $\mathbb{R}^E$. """
        return self.q_a.rsample(
            [feat.shape[1]]
        ).transpose(1, 0)

    def _rsample_noise_rec(self, graph, feat):
        """ Sample from a distribution on $\mathbb{R}^{E \times C}$. """
        return self.q_a.rsample()

    def kl_divergence(self):
        if not self.vi:
            return 0.0

        # edge_weight_sample = self._edge_weight_sample

        # kl_divergence = self.q_a.log_prob(edge_weight_sample).mean()\
        #    - self.p_a.log_prob(edge_weight_sample).mean()

        kl_divergence = torch.distributions.kl_divergence(
            self.q_a.base_distribution,
            self.p_a.base_distribution,
        )

        return kl_divergence

class FeatOnlyLayer(torch.nn.Module):
    def __init__(self, layer):
        super(FeatOnlyLayer, self).__init__()
        self.layer = layer

    def forward(self, graph, feat):
        return self.layer(feat)

class SumNodes(torch.nn.Module):
    def __init__(self, name="to_sum"):
        super(SumNodes, self).__init__()
        self.name = name

    def forward(self, graph, feat):
        graph = graph.local_var()
        graph.ndata[self.name] = feat
        feat = dgl.sum_nodes(graph, self.name)
        return feat
