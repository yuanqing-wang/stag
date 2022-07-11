import abc
import torch

class Likelihood(torch.nn.Module, abc.ABC):
    def __init__(self, distribution: type(torch.distributions.Distribution)):
        super(Likelihood, self).__init__()
        self.distribution = distribution

    @abc.abstractmethod
    def condition(self, feat):
        raise NotImplementedError

    def log_prob(self, feat, y):
        posterior = self.condition(feat)
        log_prob = posterior.log_prob(y)
        return log_prob

class CategoricalLikelihood(Likelihood):
    def __init__(self):
        super(CategoricalLikelihood, self).__init__(
            distribution=torch.distributions.Categorical
        )

    def condition(self, feat):
        return self.distribution(
            probs=feat,
        )

class BernoulliLikelihood(Likelihood):
    def __init__(self):
        super(BernoulliLikelihood, self).__init__(
            distribution=torch.distributions.Bernoulli
        )

    def condition(self, feat):
        return self.distribution(
            probs=feat,
        )
