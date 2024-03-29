import pytest

def test_import():
    import stag
    import stag.layers

def test_convert_layer():
    import stag
    import dgl
    layer = stag.zoo.GCN(16, 32)
    layer = stag.layers.StagLayer(layer)

def test_forward_r1():
    import torch, dgl
    import stag
    layer = stag.zoo.GCN(16, 32)
    layer = stag.layers.StagLayer(layer)
    g = dgl.rand_graph(3, 9)
    h = torch.randn(3, 16)
    h = layer(g, h)
    assert h.shape == torch.Size([3, 32])

def test_forward_rc():
    import torch, dgl
    import stag
    layer = stag.zoo.GCN(16, 32)
    q_a = torch.distributions.Normal(torch.ones(16), torch.ones(16))
    layer = stag.layers.StagLayer(layer, q_a=q_a)
    g = dgl.rand_graph(3, 9)
    h = torch.randn(3, 16)
    h = layer(g, h)
    assert h.shape == torch.Size([3, 32])

def test_forward_re():
    import torch, dgl
    import stag
    layer = stag.zoo.GCN(16, 32)
    q_a = stag.distributions.AmortizedDistribution(16, 1)
    layer = stag.layers.StagLayer(layer, q_a=q_a)
    g = dgl.rand_graph(3, 9)
    h = torch.randn(3, 16)
    h = layer(g, h)
    assert h.shape == torch.Size([3, 32])

def test_forward_rec():
    import torch, dgl
    import stag
    layer = stag.zoo.GCN(16, 32)
    q_a = stag.distributions.AmortizedDistribution(16, 16)
    layer = stag.layers.StagLayer(layer, q_a=q_a)
    g = dgl.rand_graph(3, 9)
    h = torch.randn(3, 16)
    h = layer(g, h)
    assert h.shape == torch.Size([3, 32])

def test_contrastive():
    import torch, dgl
    import stag
    layer = stag.zoo.GCN(16, 32)
    q_a = stag.distributions.AmortizedDistribution(16, 16)
    g = dgl.rand_graph(3, 9)
    h = torch.randn(3, 16)
    nll_contrastive = stag.layers.nll_contrastive(q_a, g, h)
    print(nll_contrastive)
