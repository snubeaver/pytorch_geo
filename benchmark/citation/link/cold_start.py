import argparse
import torch
import torch.nn.functional as F
from gat_conv import GATConv
from torch.nn import Linear
from datasets import get_planetoid_dataset
from snap_dataset import SNAPDataset
from suite_sparse import SuiteSparseMatrixCollection
from train_eval_cs import run
import pdb
from torch_geometric.nn import GCNConv
from torch.nn import Parameter
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=True)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset, num_features , num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv2 = GCNConv(128, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, data, pos_edge_index, neg_edge_index,  edge_index):

        x = F.relu(self.conv1(data.x, edge_index))
        x = self.conv2(x, pos_edge_index)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        # ef, ef -> e = maybe inner product?
        return F.log_softmax(x, dim=1), torch.einsum("ef,ef->e", x_i, x_j)
# ego-Facebook com-Amazon ego-gplus ego-twitter


# name = "ego-Facebook"
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
# dataset = SNAPDataset(path, name)
# dataset = dataset.shuffle()
# name = "{}\n------\n".format(str(dataset)[:-2])
# print(name)
# run(dataset, Net(dataset, dataset[0].num_feature, int(dataset[0].num_class)), args.runs, args.epochs, args.lr, args.weight_decay,
#         args.early_stopping)


# name = "deezer-europe"
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
# dataset = SNAPDataset(path, name)
# dataset = dataset.shuffle()
# name = "{}\n------\n".format(str(dataset)[:-2])
# print(name)
# run(dataset, Net(dataset, dataset[0].num_feature, int(dataset[0].num_class)), args.runs, args.epochs, args.lr, args.weight_decay,
#         args.early_stopping)


# names = ["Cora", "CiteSeer", "PubMed"]
names = ["Cora"]
for name in names:
    dataset = get_planetoid_dataset(name, args.normalize_features)
    name = "{}\n------\n".format(str(dataset)[:-2])
    print(name)
    run(dataset, Net(dataset, dataset.num_features, dataset.num_classes), args.runs, args.epochs, args.lr, args.weight_decay,
        args.early_stopping)
    
