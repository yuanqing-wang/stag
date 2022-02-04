import math
import torch
import dgl
from typing import Union


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

    edge_weight_distribution : torch.distributions.Distribution
        Edge weight distribution.

    Methods
    -------
    forward(graph, feat)
        Forward pass.

    """
    def __init__(
        self,
        base_layer: torch.nn.Module,
        edge_weight_distribution: \
        torch.distributions.Distribution=torch.distributions.Normal(1.0, 1.0),
        norm: bool=False,
        relu: bool=False,
    ) -> None:
        super(StagLayer, self).__init__()

        # assertions
        assert edge_weight_distribution.event_shape == torch.Size([])

        self.base_layer = base_layer

        # re-initialize edge weight distribution
        edge_weight_distribution_parameter_names\
            = list(edge_weight_distribution.arg_constraints.keys())

        edge_weight_distribution_parameters = {
            key: getattr(edge_weight_distribution, key)
            for key in edge_weight_distribution_parameter_names
            if hasattr(edge_weight_distribution, key)
        }

        for key, value in edge_weight_distribution_parameters.items():
            self.register_buffer(key, torch.tensor(value))

        self.edge_weight_distribution_instance = edge_weight_distribution.__class__
        self.edge_weight_distribution_parameters = edge_weight_distribution_parameters

        self.norm = norm
        self.relu = relu

    @property
    def edge_weight_distribution(self):
        return self.edge_weight_distribution_instance(
            **{
                key:getattr(self, key)
                for key in self.edge_weight_distribution_parameters.keys()
            }
        )

    def forward(self, graph, feat):
        """ Forward pass. """
        graph = graph.local_var()
        # rsample noise
        edge_weight_sample = self.rsample_noise(graph, feat)

        if self.relu:
            edge_weight_sample = edge_weight_sample.relu()

        self._edge_weight_sample = edge_weight_sample

        # normalize so that for each node the sum of in_degrees are the same
        if self.norm:
            edge_weight_sample = _in_norm(
                graph, edge_weight_sample,
            )

        return self.base_layer.forward(
            graph=graph,
            feat=feat,
            edge_weight=edge_weight_sample,
        )

    @property
    def q_a(self):
        """ Noise posterior. """
        return self.edge_weight_distribution

    @property
    def p_a(self):
        """ Noise prior. """
        return self.edge_weight_distribution

    def rsample_noise(self, graph, feat):
        batch_shape = self.edge_weight_distribution.batch_shape
        if batch_shape == torch.Size([]):
            edge_weight_sample = self._rsample_noise_r1(graph, feat)
        elif batch_shape == feat.shape:
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
        return self.edge_weight_distribution.rsample(
            [graph.number_of_edges(), feat.shape[1]],
        )

    def _rsample_noise_rc(self, graph, feat):
        """ Sample from a distribution on $\mathbb{R}^C$. """
        return self.edge_weight_distribution.rsample(
            [graph.number_of_edges()],
        )

    def _rsample_noise_re(self, graph, feat):
        """ Sample from a distribution on $\mathbb{R}^E$. """
        return self.edge_weight_distribution.rsample(
            [feat.shape[1]]
        ).transpose(1, 0)

    def _rsample_noise_rec(self, graph, feat):
        """ Sample from a distribution on $\mathbb{R}^{E \times C}$. """
        return self.edge_weight_distribution.rsample()


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

class StagMeanFieldVariationalInferenceLayer(StagLayer):
    """ Variational Inference layer with STAG.

    Parameters
    ----------
    base_layer : torch.nn.Module
        The basic DGL graph conv layer.

    p_a : torch.distributions.Distribution
        Prior.

    q_a_mu_init : Union[float, torch.Tensor, torch.distributions.Distribution]
        Initial mu for variational posterior.

    q_a_log_sigma_init : Union[
            float, torch.Tensor, torch.distributions.Distribution]
        Initial log_sigma for variational posterior.

    Methods
    -------
    kl_divergence(edge_weight_sample=None)
        Compute KL divergence based on a sample of edge weight.

    """
    def __init__(
            self,
            base_layer: torch.nn.Module,
            p_a: torch.distributions.Distribution\
                =torch.distributions.Normal(1.0, 1.0),
            q_a_mu_init: Union[
                    float, torch.Tensor, torch.distributions.Distribution
                ]=1.0,
            q_a_log_sigma_init: Union[
                    float, torch.Tensor, torch.distributions.Distribution
                ]=math.log(1.0),
        ):
        super(StagMeanFieldVariationalInferenceLayer, self).__init__(
            base_layer=base_layer,
        )
        if isinstance(q_a_mu_init, torch.distributions.Distribution):
            q_a_mu_init = q_a_mu_init.sample()
        if isinstance(q_a_log_sigma_init, torch.distributions.Distribution):
            q_a_log_sigma_init = q_a_log_sigma_init.sample()

        self._p_a = p_a
        self.q_a_mu = torch.nn.Parameter(torch.tensor(q_a_mu_init))
        self.q_a_log_sigma = torch.nn.Parameter(
            torch.tensor(q_a_log_sigma_init)
        )

    @property
    def q_a(self):
        return torch.distributions.Normal(
            loc=self.q_a_mu,
            scale=self.q_a_log_sigma.exp(),
        )

    @property
    def p_a(self):
        return self._p_a

    def kl_divergence(
            self,
            edge_weight_sample: Union[torch.Tensor, None]=None
        ):
        if edge_weight_sample is None:
            edge_weight_sample = self._edge_weight_sample

        kl_divergence = self.q_a.log_prob(edge_weight_sample).sum()\
            - self.p_a.log_prob(edge_weight_sample).sum()
        return kl_divergence

class StagInductiveMeanFieldVariationalInferenceLayer(StagLayer):
    """ Variational Inference layer with STAG.

    Parameters
    ----------
    base_layer : torch.nn.Module
        The basic DGL graph conv layer.

    p_a : torch.distributions.Distribution
        Prior.

    q_a_mu_init : Union[float, torch.Tensor, torch.distributions.Distribution]
        Initial mu for variational posterior.

    q_a_log_sigma_init : Union[
            float, torch.Tensor, torch.distributions.Distribution]
        Initial log_sigma for variational posterior.

    Methods
    -------
    kl_divergence(edge_weight_sample=None)
        Compute KL divergence based on a sample of edge weight.

    """
    def __init__(
            self,
            base_layer: torch.nn.Module,
            p_a: torch.distributions.Distribution\
                =torch.distributions.Normal(1.0, 1.0),
        ):
        super(StagInductiveMeanFieldVariationalInferenceLayer, self).__init__(
            base_layer=base_layer,
        )
