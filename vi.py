import torch
import dgl

class StagVIRNodeClassifiction(torch.nn.Module):
    def __init__(
            self,
            layer,
            in_features,
            hidden_features,
            out_features,
            depth,
            activation=None,
            a_prior=1.0,
            a_log_sigma_init=1.0,
            kl_scaling=1.0,
        ):
        super(StagVIRNodeClassifiction, self).__init__()

        # get activation function from pytorch if specified
        if activation is not None:
            activation = getattr(torch.nn.functional, activation)

        # initial layer: in -> hidden
        self.gn0 = layer(
            in_feats=in_features,
            out_feats=hidden_features,
            activation=activation,
        )

        self.bn0 = torch.nn.BatchNorm1d(hidden_features)

        # last layer: hidden -> out
        setattr(
            self,
            "gn%s" % (depth-1),
            layer(
                in_feats=hidden_features,
                out_feats=hidden_features,
                activation=None,
            )
        )

        # middle layers: hidden -> hidden
        for idx in range(1, depth-1):
            setattr(
                self,
                "gn%s" % idx,
                layer(
                    in_feats=hidden_features,
                    out_feats=hidden_features,
                    activation=activation
                )
            )

            setattr(
                self,
                "bn%s" % idx,
                torch.nn.BatchNorm1d(hidden_features),
            )

        self.a_prior = torch.distributions.Normal(1.0, a_prior)

        # middle layers
        self.a_mu = torch.nn.Parameter(torch.tensor(1.0))

        self.a_log_sigma = torch.nn.Parameter(
            torch.tensor(a_log_sigma_init)
        )

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.kl_scaling = kl_scaling
        self.activation = activation


    def _forward(self, g, x):
        """ Internal forward.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph.

        x : torch.Tensor
            Input features. (n_nodes, in_features)

        Returns
        -------
        dgl.DGLGraph
            Graph with perturbed weights annotated.

        """
        _g = g.local_var()

        _g.apply_edges(
            lambda edges: {
                **{
                    "a0": torch.distributions.Normal(
                        self.a_mu, self.a_log_sigma,
                        ).rsample([_g.number_of_edges(), self.in_features]),
                },
                **{
                    "a%s" % idx: torch.distributions.Normal(
                        self.a_mu, self.a_log_sigma,
                        ).rsample([_g.number_of_edges(), self.hidden_features])
                    for idx in range(1, self.depth)
                }
            }
        )

        for idx in range(self.depth):
            _g.edata["a"] = _g.edata["a%s" % idx]
            x = getattr(self, "gn%s" % idx)(g, x)

            if idx != self.depth - 1:
                # x = getattr(self, "bn%s" % idx)(x)
                x = self.activation(x)

        _g.ndata["x"] = x
        return _g, x

    @staticmethod
    def condition(x):
        """ Condition (static method. )

        Takes a set of parameters and return a distribution.

        Parameters
        ----------
        x : torch.Tensor (n_graphs, out_features)

        Returns
        -------
        torch.distributions.Categorical

        """
        return torch.distributions.categorical.Categorical(logits=x)

    def loss(self, g, x, y, n_samples=1, mask=None):
        """ Training loss.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph.

        x : torch.Tensor
            Input features.

        y : torch.Tensor
            Reference value.

        n_samples : int
            Number of samples to consider.

        Returns
        -------
        torch.Tensor : loss

        """

        # initialize losses
        losses = []

        for _ in range(n_samples):
            _x = x
            # get input graph
            # x.shape = (n_graphs, out_features)
            _g, _x = self._forward(g, _x)
            _x = _x[mask]

            # posterior distribution
            p_y_given_x_z = self.condition(_x)

            # prior
            p_a = self.a_prior

            # variational posterior
            q_a = torch.distributions.Normal(
                loc=self.a_mu,
                scale=self.a_log_sigma.exp()
            )

            # compute elbo
            elbo = p_y_given_x_z.log_prob(y[mask]).sum()\
                + sum(
                    [
                        p_a.log_prob(_g.edata["a%s" % idx]).sum()\
                        - q_a.log_prob(_g.edata["a%s" % idx]).sum()
                        for idx in range(self.depth)
                    ]
                )

            losses.append(-elbo)

        return sum(losses) / n_samples

    def forward(self, g, x, n_samples=1, mask=None):
        """ Forward pass. (Inference Pass)

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph.

        x : torch.Tensor
            Input features.

        n_samples : int
            Number of samples to consider.
        """
        _g = g.local_var()

        if mask is None:
            x = torch.stack(
                [self._forward(_g, x)[1].softmax(dim=-1) for _ in range(n_samples)], dim=0
            ).mean(dim=0)
        else:
            x = torch.stack(
                    [self._forward(_g, x)[1][mask].softmax(dim=-1) for _ in range(n_samples)], dim=0
            ).mean(dim=0)
        return x
