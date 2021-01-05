import torch
import dgl
import stag
        
class Net(torch.nn.Module):
    def __init__(
            self, 
            layer, 
            in_features, 
            hidden_features, 
            out_features, 
            depth, 
            activation=None
        ):
        super(Net, self).__init__()
        
        # local import
        from dgl.nn import pytorch as dgl_nn

        # get activation function from pytorch if specified
        if activation is not None:
            activation = getattr(torch.nn.functional, activation)

        # initial layer: in -> hidden
        self.gn0 = getattr(dgl_nn, layer)(
            in_feats=in_features, 
            out_feats=hidden_features,
            activation=activation,
        )

        # last layer: hidden -> out
        setattr(
            self, 
            "gn%s" % (depth-1), 
            getattr(dgl_nn ,layer)(
                in_feats=hidden_features,
                out_feats=out_features,
            )
        )

        # middle layers: hidden -> hidden
        for idx in range(1, depth-2):
            setattr(
                self,
                "gn%s" % idx,
                getattr(
                    in_feats=hidden_features,
                    out_feats=hidden_features,
                    activation=activation
                )
            )

        self.depth = depth
        
    def forward(self, g, x):
        # ensure local scope
        g = g.local_var()

        for idx in range(self.depth):
            x = getattr(self, "gn%s" % idx)(g, x)

        return x



