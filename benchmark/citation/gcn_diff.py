import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from gcn_conv_high import GCNConv_, GCNConv_random

from datasets import get_planetoid_dataset
from train_eval_diff import random_planetoid_splits, run
import pdb
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()

args.use_gdc=True
class Net(torch.nn.Module):
    def __init__(self, data, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
# pdb.set_trace()
permute_masks = random_planetoid_splits if args.random_splits else None
data = dataset[0]
gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128,
                                        dim=0), exact=True)
data = gdc(data)
# pdb.set_trace()
# data.num_classes = dataset[0].num_classes
# dataset = get_planetoid_dataset(args.dataset, args.normalize_features)

run(dataset, data, Net(data, dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
