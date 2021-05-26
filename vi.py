import torch
import dgl
class StagVI(torch.nn.Module):
    def __init__(
            self,
            layer,
            in_features,
            hidden_features,
            out_features,
            depth,
            activation=None,
            a_prior=1.0,
            a_mu_init_std=1.0,
            a_log_sigma_init=1.0,
            kl_scaling=1.0,
        ):
        super(StagVI, self).__init__()

        # get activation function from pytorch if specified
        if activation is not None:
            activation = getattr(torch.nn.functional, activation)
        self.activation = activation

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

        if depth == 1:
            self.gn0 = layer(
                in_feats=in_features, out_feats=out_features, activation=None
            )

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.kl_scaling = kl_scaling
        self.a_prior = torch.distributions.Normal(1.0, a_prior)
        self.a_mu_init_std = a_mu_init_std
        self.a_log_sigma_init = a_log_sigma_init

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

    def q_a_array(self, g):
        return [
            torch.distributions.Normal(
                loc=self.a_mu,
                scale=self.a_log_sigma.exp()
            )
            for _ in range(self.depth)
        ]

    def loss(self, g, x, y, n_samples=1, mask=None, edge_subsample=1.0):
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
        neg_elbos = []

        xs = []

        for _ in range(n_samples):
            _g = g.local_var()
            _g, _x = self._forward(_g, x)

            # posterior distribution
            p_y_given_x_z = self.condition(_x)

            # prior
            p_a = self.a_prior

            # variational posterior
            q_a_array = self.q_a_array(_g)

            # compute edge regularizer
            _g.apply_edges(
                lambda edge: {
                    "reg": -sum(
                        [
                            p_a.log_prob(edge.data["a%s" % idx]).sum(dim=-1)
                            - q_a_array[idx].log_prob(edge.data["a%s" % idx]).sum(dim=-1)\
                            for idx in range(self.depth)
                        ]
                    )
                }
            )

            aggregate_fn = dgl.function.copy_e('reg', 'm_reg')

            # mean averages the regularization term
            _g.update_all(aggregate_fn, dgl.function.mean(msg='m_reg', out='reg'))

            neg_elbo_z = -p_y_given_x_z.log_prob(y)[mask].sum()\
                + self.kl_scaling * _g.ndata["reg"][mask].sum()

            neg_elbos.append(neg_elbo_z)

        return sum(neg_elbos) / n_samples


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


class StagVI_NodeClassification_R1(StagVI):
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
        super(StagVI_NodeClassification_R1, self).__init__(
            layer=layer,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            activation=activation,
            a_prior=a_prior,
            a_log_sigma_init=a_log_sigma_init,
            kl_scaling=kl_scaling,
        )

        self.a_prior = torch.distributions.Normal(1.0, a_prior)

        # middle layers
        self.a_mu = torch.nn.Parameter(torch.tensor(1.0))

        self.a_log_sigma = torch.nn.Parameter(
            torch.tensor(a_log_sigma_init)
        )


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
                x = self.activation(x)

        _g.ndata["x"] = x
        return _g, x

class StagVI_NodeClassification_RC(StagVI):
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
            a_mu_init_std=1.0,
            kl_scaling=1.0,
        ):
        super(StagVI_NodeClassification_RC, self).__init__(
            layer=layer,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            activation=activation,
            a_prior=a_prior,
            a_log_sigma_init=a_log_sigma_init,
            kl_scaling=kl_scaling,
        )

        self.a_prior = torch.distributions.Normal(1.0, a_prior)

        # first layer
        self.a_mu_0 = torch.nn.Parameter(
            torch.distributions.Normal(1, a_mu_init_std).sample(
                [in_features]
            )
        )

        self.a_log_sigma_0 = torch.nn.Parameter(
            a_log_sigma_init * torch.ones(in_features)
        )

        for idx in range(1, self.depth):
            setattr(
                self,
                "a_mu_%s" % idx,
                torch.nn.Parameter(
                    torch.distributions.Normal(1, a_mu_init_std).sample(
                        [hidden_features]
                    )
                )
            )

            setattr(
                self,
                "a_log_sigma_%s" % idx,
                torch.nn.Parameter(
                    a_log_sigma_init * torch.ones([hidden_features])
                )
            )

    def q_a_array(self, g):
        return [
            torch.distributions.Normal(
                loc=getattr(self, "a_mu_%s" % idx),
                scale=getattr(self, "a_log_sigma_%s" % idx).exp()
            )
            for idx in range(self.depth)
        ]

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
                "a%s" % idx: torch.distributions.Normal(
                    getattr(self, "a_mu_%s" % idx),
                    getattr(self, "a_log_sigma_%s" % idx).exp(),
                ).rsample([_g.number_of_edges()])
                for idx in range(self.depth)
            }
        )

        for idx in range(self.depth):
            _g.edata["a"] = _g.edata["a%s" % idx]
            x = getattr(self, "gn%s" % idx)(_g, x)

            if idx != self.depth - 1:
                x = self.activation(x)

        _g.ndata["x"] = x

        return _g, x


class StagVI_NodeClassification_RE(StagVI):
    def __init__(
            self,
            layer,
            in_features,
            hidden_features,
            out_features,
            depth,
            activation=None,
            a_prior=1.0,
            kl_scaling=1.0,
            a_mu_init_std=1.0,
            a_log_sigma_init=0.0,
        ):
        super(StagVI_NodeClassification_RE, self).__init__(
            layer=layer,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            activation=activation,
            a_prior=a_prior,
            a_log_sigma_init=a_log_sigma_init,
            kl_scaling=kl_scaling,
        )

        self.gn0_enc = layer(in_feats=in_features, out_feats=hidden_features, activation=self.activation)
        self.gn1_enc = layer(in_feats=hidden_features, out_feats=hidden_features, activation=self.activation)

        self.f_z_mu = torch.nn.Linear(2*hidden_features, 1)
        self.f_z_log_sigma = torch.nn.Linear(2*hidden_features, 1)

        torch.nn.init.normal_(self.f_z_log_sigma.weight, std=1e-3)
        torch.nn.init.constant_(self.f_z_log_sigma.bias, a_log_sigma_init)

        torch.nn.init.normal_(self.f_z_mu.weight, std=1e-2)
        torch.nn.init.normal_(self.f_z_mu.bias, mean=1.0, std=a_mu_init_std)

    def q_a(self, g):
        return torch.distributions.Normal(
            g.edata["mu"], g.edata["sigma"]
        )


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

        z_node = self.gn0_enc(_g, x)
        z_node = self.gn1_enc(_g, z_node)
        _g.ndata["z_node"] = z_node

        _g.apply_edges(lambda edges:{
            **{
                "mu": self.f_z_mu(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1))
            },
            **{
                "sigma": self.f_z_log_sigma(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).exp()
            },
        }
        )

        _g.apply_edges(
            lambda edges: {
                "a%s" % idx: torch.distributions.Normal(
                    loc=edges.data["mu"],
                    scale=edges.data["sigma"],
                ).rsample()
                for idx in range(self.depth)
            }
        )

        for idx in range(self.depth):
            _g.edata["a"] = _g.edata["a%s" % idx]
            x = getattr(self, "gn%s" % idx)(g, x)

            if idx != self.depth - 1:
                x = self.activation(x)

        _g.ndata["x"] = x
        return _g, x


class StagVI_NodeClassification_REC(StagVI):
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
            a_mu_init_std=1.0,
            kl_scaling=1.0,
        ):
        super(StagVI_NodeClassification_REC, self).__init__(
            layer=layer,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            activation=activation,
            a_prior=a_prior,
            a_log_sigma_init=a_log_sigma_init,
            kl_scaling=kl_scaling,
        )

        self.a_prior = torch.distributions.Normal(1.0, a_prior)

        self.gn0_enc = layer(in_feats=in_features, out_feats=hidden_features, activation=self.activation)
        self.gn1_enc = layer(in_feats=hidden_features, out_feats=hidden_features, activation=self.activation)

        for idx in range(self.depth):
            if idx == 0:
                _hidden_features = in_features
            else:
                _hidden_features = hidden_features

            setattr(self, "f_z_mu_%s" % idx, torch.nn.Linear(2 * hidden_features, _hidden_features))
            setattr(self, "f_z_log_sigma_%s" % idx, torch.nn.Linear(2 * hidden_features, _hidden_features))

            # torch.nn.init.constant_(
            #     getattr(self, "f_z_mu_%s" % idx).bias,
            #     1.0
            # )

            # torch.nn.init.constant_(
            #     getattr(self, "f_z_log_sigma_%s" % idx).bias,
            #     a_log_sigma_init,
            # )

        self.a_mean_history = []
        self.a_std_history = []

    def q_a_array(self, g):
        return [
            torch.distributions.Normal(
                loc=g.edata["mu_%s" % idx],
                scale=g.edata["sigma_%s" % idx],
            )
            for idx in range(self.depth)
        ]

    def contrastive_loss(self, g):
        _g = g.local_var()
        fake_src_idxs = torch.multinomial(torch.ones(_g.number_of_nodes()), _g.ndata["train_mask"].sum(), replacement=True)
        fake_dst_idxs = torch.multinomial(torch.ones(_g.number_of_nodes()), _g.ndata["train_mask"].sum(), replacement=True)

        # fake_src_idxs = torch.randperm(_g.number_of_nodes())
        # fake_dst_idxs = torch.randperm(_g.number_of_nodes())

        return -sum(
            [
                torch.distributions.Normal(
                    loc=getattr(self, "f_z_mu_%s" % idx)(torch.cat([_g.ndata["z_node"][fake_src_idxs, :], _g.ndata["z_node"][fake_dst_idxs, :]], dim=-1)),
                    scale=getattr(self, "f_z_log_sigma_%s" % idx)(torch.cat([_g.ndata["z_node"][fake_src_idxs, :], _g.ndata["z_node"][fake_dst_idxs, :]], dim=-1)).exp(),
                ).log_prob(torch.tensor(0.0)).sum()
                for idx in range(self.depth)
            ]
        )
        - sum(
            [
                torch.distributions.Normal(
                    loc=g.edata["mu_%s" % idx],
                    scale=g.edata["sigma_%s" % idx],
                ).log_prob(torch.tensor(1.0)).sum()
                for idx in range(self.depth)
            ]
        )

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

        z_node = self.gn0_enc(_g, x)
        z_node = self.gn1_enc(_g, z_node)
        _g.ndata["z_node"] = z_node

        _g.apply_edges(lambda edges:{
            **{
                "mu_%s" % idx: getattr(self, "f_z_mu_%s" % idx)(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).exp()
                for idx in range(self.depth)
            },
            **{
                "sigma_%s" % idx: getattr(self, "f_z_log_sigma_%s" % idx)(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).exp()
                for idx in range(self.depth)
            }
        }
        )

        _g.apply_edges(
            lambda edges: {
                "a%s" % idx: torch.distributions.Normal(
                    loc=edges.data["mu_%s" % idx],
                    scale=edges.data["sigma_%s" % idx],
                ).rsample()
                for idx in range(self.depth)
            }
        )

        for idx in range(self.depth):
            _g.edata["a"] = _g.edata["a%s" % idx]
            x = getattr(self, "gn%s" % idx)(_g, x)

        _g.ndata["x"] = x
        return _g, x
