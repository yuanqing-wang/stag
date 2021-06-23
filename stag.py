import torch
from dataset import Dataset
import vi
import dgl

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
        # sample edge weight
        event_shape = self.edge_weight_distribution.event_shape

        # rsample noise
        noise = self.rsample_noise(graph, feat)

        return self.base_layer.forward(
            graph=graph,
            feat=feat,
            edge_weight=noise,
        )

    def q_a(self, noise):
        


    def rsample_noise(self, graph, feat):
        noise = {
            torch.Size([]): self._rsample_noise_r1(graph, feat),
            feat.shape: self._rsample_noise_rc(graph, feat),
            torch.Size([graph.number_of_edges()]):\
                self._rsample_noise_re(graph, feat),
            torch.Size([graph.number_of_edges(), feat.shape[1]]):\
                self._rsample_noise_rec(graph, feat)
        }[self.edge_weight_distribution.event_shape]

        event_shape = self.edge_weight_distribution.event_shape

        if event_shape == torch.Size([]):
            return self._rsample_noise_r1(graph, feat)
        elif event_shape == feat.shape:
            return self._rsample_noise_rc(graph, feat)
        elif event_shape == torch.Size([graph.number_of_edges()]):
            return self._rsample_noise_re(graph, feat)
        elif event_shape == torch.Size(
            [graph.number_of_edges(), feat.shape[1]]
        ):
            return self._rsample_noise_rec(graph, feat)

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
