import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)
        # num_node 최대 개수로
        num_nodes = max_num_nodes
        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        
        dataset = dataset[torch.tensor(indices)]


        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset

class MyDenseCollater(object):
    def collate(self, data_list):
        batch = Batch()
        for key in data_list[0].keys:
            if key=='adj':
                adjlist=[]
                fvlist=[]
                fv2list=[]
                for d in data_list:
                    symadj = ((d[key] + d[key].T) > 0).to(torch.float)
                    D = symadj.sum(-1)
                    D = torch.diag_embed(D)
                    L = D - symadj
                    e, v = torch.symeig(L, eigenvectors=True)
                    e[e < 1e-4] = 0.
                    fiedler_idx = (e == 0.).sum(dim=0)
                    fv = v[:, fiedler_idx:fiedler_idx+2]
                    adjlist.append(symadj)
                    fvlist.append(fv)
                batch[key] = default_collate(adjlist)
                batch['fv'] = default_collate(fvlist)
            else:
                batch[key] = default_collate([d[key] for d in data_list])

        return batch

    def __call__(self, batch):

        return self.collate(batch)
class DenseCollater(object):
    def collate(self, data_list):
        batch = Batch()
        for key in data_list[0].keys:
            batch[key] = default_collate([d[key] for d in data_list])
        return batch

    def __call__(self, batch):
        return self.collate(batch)
class MyDenseDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    .. note::

        To make use of this data loader, all graphs in the dataset needs to
        have the same shape for each its attributes.
        Therefore, this data loader should only be used when working with
        *dense* adjacency matrices.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=DenseCollater(), num_workers=0, **kwargs):
        super(MyDenseDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=collate_fn, num_workers=num_workers, **kwargs)