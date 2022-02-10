from .gcn import GCN

import dgl
from functools import partial
GraphSAGE = partial(
    dgl.nn.SAGEConv,
    aggregator_type="lstm",
)
