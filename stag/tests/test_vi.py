import pytest

def test_import():
    import stag
    import stag.layers

def test_convert_layer():
    import stag
    import dgl
    layer = dgl.nn.GraphConv(16, 32)
    layer = stag.layers.StagMeanFieldVariationalInferenceLayer(layer)

def test_forward():
    import torch, dgl
    import stag
    layer = dgl.nn.GraphConv(16, 32)
    layer = stag.layers.StagMeanFieldVariationalInferenceLayer(layer)
    g = dgl.rand_graph(3, 9)
    h = torch.randn(3, 16)
    h = layer(g, h)
    kl_divergence = layer.kl_divergence()
    assert kl_divergence == 0.0

def test_forward_rc():
    import torch, dgl
    import stag
    layer = dgl.nn.GraphConv(16, 32)
    layer = stag.layers.StagMeanFieldVariationalInferenceLayer(
        layer,
        q_a_mu_init=torch.ones(16),
        q_a_log_sigma_init=torch.distributions.Normal(
            torch.ones(16),
            torch.ones(16),
        ),
    )
    g = dgl.rand_graph(3, 9)
    h = torch.randn(3, 16)
    h = layer(g, h)
    kl_divergence = layer.kl_divergence()
