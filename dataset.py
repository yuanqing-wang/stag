# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
from typing import Union, List

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Dataset(torch.utils.data.Dataset):
    """A collection of Points with functionalities to be compatible with
    training and optimization.

    Parameters
    ----------
    points : List[Point]
        A list of points.

    Methods
    -------
    featurize(points)
        Featurize all points in the dataset.
    view()
        Generate a torch.utils.data.DataLoader from this Dataset.

    """

    def __init__(self, points) -> None:
        super(Dataset, self).__init__()
        self.points = points

    def __repr__(self):
        return "%s with %s points" % (self.__class__.__name__, len(self))

    def __len__(self):
        if self.points is None:
            return 0
        return len(self.points)

    def __getitem__(self, idx):
        if self.points is None:
            raise RuntimeError("Empty Portfolio.")
        if isinstance(idx, int):
            return self.points[idx]
        elif isinstance(idx, torch.Tensor):
            idx = idx.detach().flatten().cpu().numpy().tolist()
        if isinstance(idx, list):
            return self.__class__(points=[
                self.points[_idx] for _idx in idx
            ])

        return self.__class__(points=self.points[idx])

    def __iter__(self):
        return iter(self.points)

    @staticmethod
    def batch_of_g_and_y(points):
        # initialize results
        gs = []
        ys = []

        # loop through the points
        for point in points:
            if not point.is_featurized():  # featurize
                point.featurize()
            if point.y is None:
                raise RuntimeError("No data associated with data. ")
            gs.append(point.g)
            ys.append(point.y)

        g = dgl.batch(gs)
        y = torch.tensor(ys)[:, None]
        return g, y

    @staticmethod
    def batch_of_g(points):
        # initialize results
        gs = []

        # loop through the points
        for point in points:
            if not point.is_featurized():  # featurize
                point.featurize()
            gs.append(point.g)

        g = dgl.batch(gs)
        return g

    def view(
        self,
        collate_fn: Union[callable, str] = "batch_of_g_and_y",
        *args,
        **kwargs
    ):
        """Provide a data loader from portfolio.

        Parameters
        ----------
        collate_fn : None or callable
            The function to gather data points.

        Returns
        -------
        torch.utils.data.DataLoader
            Resulting data loader.

        """
        # provide default collate function
        if isinstance(collate_fn, str):
            collate_fn = getattr(self, collate_fn)

        return torch.utils.data.DataLoader(
            dataset=self,
            collate_fn=collate_fn,
            *args,
            **kwargs,
        )
