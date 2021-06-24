import torch

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
    ) -> None:
        super(StagLayer, self).__init__()

        # assertions
        assert edge_weight_distribution.event_shape == torch.Size([])

        self.base_layer = base_layer
        self.edge_weight_distribution = edge_weight_distribution

    def forward(self, graph, feat):
        """ Forward pass. """
        # rsample noise
        edge_weight_sample = self.rsample_noise(graph, feat)

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
