import pytest

def test_import():
    import stag
    import stag.layers
    import stag.vi

def test_convert_layer():
    import stag
    import dgl
    layer = dgl.nn.GraphConv(16, 32)
    layer = stag.vi.StagMeanFieldVariationalInference(layer)

    
