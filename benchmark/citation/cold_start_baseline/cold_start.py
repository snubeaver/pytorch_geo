import argparse
import torch
import torch.nn.functional as F
from gat_score import GATScore
from torch.nn import Linear
from datasets import get_planetoid_dataset
from train_eval_cs import random_planetoid_splits, run
import pdb
from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=True)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.score1 = GATScore(Linear(dataset.num_features*2, 1))
        self.score2 = GATScore(Linear(args.hidden*2, 1))
        self.score3 = GATScore(Linear(args.hidden*2, 1))
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, pos_edge_index, neg_edge_index):
        x, edge_index = data.x, data.edge_index
        # 0 layer
        # s1 = self.score1(x, masked_node)

        # 1 layer
        x = F.relu(self.conv1(x, edge_index))
        # masked_node = F.relu(self.conv1(masked_node, torch.zeros([2,1], dtype=edge_index.dtype, device= edge_index.device)))
        # s2 = self.score2(x, masked_node)

        # 2 layer
        x = self.conv2(x, edge_index)
        # masked_node = self.conv2(masked_node, torch.zeros([2,1], dtype=edge_index.dtype, device= edge_index.device))
        # s3 = self.score2(x, masked_node)
        
        # x[data.cold_mask_node] = masked_node
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)


        # return F.softmax((s1+s2+s3), 1)



dataset = get_planetoid_dataset(args.dataset, args.normalize_features)

permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
