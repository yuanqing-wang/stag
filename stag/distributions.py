import torch
from torch.distributions import constraints
from typing import Union, Callable
from functools import partial

class Distribution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def batch_shape(self):
        return self.base_distribution.batch_shape

    @property
    def mean(self):
        return self.base_distribution.mean

    @property
    def stddev(self):
        return self.base_distribution.stddev

    @property
    def variance(self):
        return self.base_distribution.variance

    def expand(self, *args, **kwargs):
        return self.base_distribution.expand(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        return self.base_distribution.rsample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.base_distribution.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.base_distribution.log_prob(*args, **kwargs)

    def cdf(self, *args, **kwargs):
        return self.base_distribution.cdf(*args, **kwargs)

    def icdf(self, *args, **kwargs):
        return self.base_distribution.icdf(*args, **kwargs)

    def entropy(self, *args, **kwargs):
        return self.base_distribution.entropy(*args, **kwargs)

    def condition(self, *args, **kwargs):
        return self

class DeltaDistribution(Distribution):
    def __init__(self, value=0.0):
        super().__init__()
        value = torch.tensor(value)
        self.register_buffer("value", value)

    @property
    def batch_shape(self):
        return self.value.shape

    @property
    def mean(self):
        return self.value

    @property
    def stddev(self):
        return torch.zeros_like(self.value)

    @property
    def variance(self):
        return torch.zeros_like(self.value)

    def expand(self, *args, **kwargs):
        raise NotImplementedError

    def rsample(self, *args, **kwargs):
        return self.value

    def sample(self, *args, **kwargs):
        return self.value

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError

    def cdf(self, *args, **kwargs):
        raise NotImplementedError

    def icdf(self, *args, **kwargs):
        raise NotImplementedError

    def entropy(self, *args, **kwargs):
        raise NotImplementedError

class ParametrizedDistribution(Distribution):
    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
        vi: bool=False,
    ):
        super().__init__()
        parameter_names = list(base_distribution.arg_constraints.keys())
        if "logits" in parameter_names: parameter_names.remove("logits")

        parameters = {
            key: getattr(base_distribution, key)
            for key in parameter_names
        }

        if vi:
            new_parameter_names = []
            for key, value in parameters.items():
                if base_distribution.__class__.arg_constraints[key] == constraints.positive:
                    setattr(
                        self,
                        "log_" + key,
                        torch.nn.Parameter(torch.log(torch.tensor(value))),
                    )
                    new_parameter_names.append("log_" + key)

                else:
                    setattr(self, key, torch.nn.Parameter(torch.tensor(value)))
                    new_parameter_names.append(key)

        else:
            for key, value in parameters.items():
                self.register_buffer(key, torch.tensor(value))
            new_parameter_names = parameter_names

        self.base_distribution_instance = partial(
            base_distribution.__class__,
            validate_args=False,
        )
        self.new_parameter_names = new_parameter_names

    def __repr__(self):
         return repr(self.base_distribution)

    @property
    def base_distribution(self):
        return self.base_distribution_instance(
            **{
                key.replace("log_", ""):getattr(self, key).exp() if "log_" in key else getattr(self, key)
                for key in self.new_parameter_names
            }
        )

class AmortizedDistribution(Distribution):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Union[None, int]=None,
        activation: Callable=torch.nn.SiLU(),
        base_distribution_class: type=torch.distributions.Normal,
        init_like: Union[None, torch.distributions.Distribution, Distribution]=None,
    ):
        super().__init__()
        if hidden_features is None:
            hidden_features = out_features

        n_parameters = len(base_distribution_class.arg_constraints)

        new_parameter_names = []
        parameter_names = list(base_distribution_class.arg_constraints.keys())
        for parameter_name in parameter_names:
            if base_distribution_class.arg_constraints[parameter_name] == constraints.positive:
                new_parameter_name = "log_" + parameter_name
                new_parameter_names.append(new_parameter_name)
            else:
                new_parameter_names.append(parameter_name)
        self.new_parameter_names = new_parameter_names

        self.embedding_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_features + 1, hidden_features),
            activation,
            # torch.nn.Linear(hidden_features, hidden_features),
            # activation,
        )


        self.parameters_mlp = torch.nn.ModuleDict(
            {
                key: torch.nn.Linear(hidden_features, out_features)
                for key in self.new_parameter_names
            }
        )

        self.base_distribution_class = base_distribution_class
        self.out_features = out_features

        if init_like is not None:
            self._init_like(init_like)

    def _init_like(self, init_like):
        if isinstance(init_like, Distribution):
            init_like = init_like.base_distribution

        for parameter in self.new_parameter_names:
            # torch.nn.init.normal_(
            #     self.parameters_mlp[parameter].weight,
            #     0.0, 1e-5,
            # )

            if "log_" in parameter:
                torch.nn.init.constant_(
                    self.parameters_mlp[parameter].bias,
                    torch.log(getattr(init_like, parameter.replace("log_", ""))),
                )

            else:
                torch.nn.init.constant_(
                    self.parameters_mlp[parameter].bias,
                    getattr(init_like, parameter),
                )

    def condition(self, graph, feat):
        graph = graph.local_var()
        graph.ndata['h'] = feat
        graph.ndata['id'] = graph.nodes().unsqueeze(-1)

        graph.apply_edges(
            lambda edges: {'h': self.embedding_mlp(torch.cat([edges.src['h'], edges.dst['h'], 1.0*(edges.src['id'] == edges.dst['id'])], dim=-1))},
        )

        self.new_parameters = dict(
            {key: self.parameters_mlp[key](graph.edata['h']) for key in self.new_parameter_names},
        )

        return self

    @property
    def base_distribution(self):
        return self.base_distribution_class(
            **{
                key.replace("log_", ""):self.new_parameters[key].exp() if "log_" in key else self.new_parameters[key]
                for key in self.new_parameter_names
            }
        )
