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
        torch.ones_like(current_sum),
    )

    graph.ndata["s"] = node_scaling

    # put scaling back to edges
    graph.apply_edges(lambda edges: {"s": edges.dst["s"]})
    edge_scaling = graph.edata["s"]
    edge_weight_sample = edge_weight_sample * edge_scaling
    return edge_weight_sample

def nll_contrastive(q_a, graph, feat):
    graph = graph.local_var()
    graph.ndata['h'] = feat

    # positive edge distribution
    graph.apply_edges(
        lambda edges: {'h': q_a.embedding_mlp(torch.cat([edges.src['h'], edges.dst['h']], dim=-1))},
    )

    new_parameters = {key: q_a.parameters_mlp[key](graph.edata['h']) for key in q_a.new_parameter_names}
    q_a_positive = q_a.base_distribution_class(
            **{
                key.replace("log_", ""): new_parameters[key].exp() if "log_" in key else new_parameters[key]
                for key in q_a.new_parameter_names
            }
    )

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

    nll = -q_a_positive.log_prob(torch.tensor(1.0, device=feat.device)).mean() - q_a_negative.log_prob(torch.tensor(0.0, device=feat.device)).mean()
    return nll

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

        self.add_module("q_a", q_a)

        self.p_a = p_a
        self.norm = norm
        self.relu = relu
        self.vi = vi

    def forward(self, graph, feat):
        """ Forward pass. """
        graph = graph.local_var()

        self.q_a.condition(graph, feat)

        if hasattr(self.base_layer, "sample_dimension"):
            sample_dimension = self.base_layer.sample_dimension
        else:
            sample_dimension = feat.shape[-1]

        # rsample noise
        edge_weight_sample = self.rsample_noise(graph, sample_dimension)

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

    def rsample_noise(self, graph, sample_dimension):

        edge_weight_sample = self.q_a.expand(
            [graph.number_of_edges(), sample_dimension],
        ).rsample()

        return edge_weight_sample


    def kl_divergence(self):
        if not self.vi:
            return 0.0

        # edge_weight_sample = self._edge_weight_sample

        # kl_divergence = self.q_a.log_prob(edge_weight_sample).mean()\
        #    - self.p_a.log_prob(edge_weight_sample).mean()

        kl_divergence = torch.distributions.kl_divergence(
            self.q_a.base_distribution,
            self.p_a.base_distribution,
        ).mean()

        return kl_divergence


class StagLayerContrastive(StagLayer):
    def nll_contrastive(self, graph, feat):
        """ Forward pass. """
        graph = graph.local_var()
        # negative edge distribution
        fake_src = torch.randint(high=graph.number_of_nodes(), size=[graph.number_of_edges()])
        fake_dst = torch.randint(high=graph.number_of_nodes(), size=[graph.number_of_edges()])
        h_fake = self.q_a.embedding_mlp(torch.cat([graph.ndata['h'][fake_src], graph.ndata['h'][fake_dst]], dim=-1))
        fake_new_parameters = {key: self.q_a.parameters_mlp[key](h_fake) for key in self.q_a.new_parameter_names}
        q_a_negative = self.q_a.base_distribution_class(
                **{
                    key.replace("log_", ""): fake_new_parameters[key].exp() if "log_" in key else fake_new_parameters[key]
                    for key in self.q_a.new_parameter_names
                }
        )

        nll = -self.q_a.log_prob(torch.tensor(1.0, device=feat.device)).mean() - q_a_negative.log_prob(torch.tensor(0.0, device=feat.device)).mean()
        return nll

class FeatOnlyLayer(torch.nn.Module):
    vi = False
    def __init__(self, layer):
        super(FeatOnlyLayer, self).__init__()
        self.layer = layer

    def forward(self, graph, feat):
        return self.layer(feat)

class SumNodes(torch.nn.Module):
    vi = False
    def __init__(self, name="to_sum"):
        super(SumNodes, self).__init__()
        self.name = name

    def forward(self, graph, feat):
        graph = graph.local_var()
        graph.ndata[self.name] = feat
        feat = dgl.sum_nodes(graph, self.name)
        return feat
