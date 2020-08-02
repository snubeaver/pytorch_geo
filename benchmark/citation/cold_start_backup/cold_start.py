import argparse
import torch
import torch.nn.functional as F
from gat_score import GATScore
from torch.nn import Linear
from datasets import get_planetoid_dataset
from train_eval_cs import run
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
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        # self.score = GATScore(Linear(args.hidden*2, 1))
        self.score1 = Linear(dataset.num_features, 1)
        self.score2 = Linear(args.hidden, 1)
        self.score3 = Linear(dataset.num_classes, 1)
        self.sum = Linear(3,1)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.score1.reset_parameters()
        self.score2.reset_parameters()
        self.score3.reset_parameters()


    def forward(self, data, pos_edge_index, neg_edge_index, edge_index):
        x,  masked_nodes = data.x,  data.masked_nodes
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])

        dist1 = x_j-x_i
        o1 = F.relu(self.score1(dist1).squeeze())

        x = F.relu(self.conv1(x, edge_index))

        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])

        dist2 = x_j-x_i
        o2 = F.relu(self.score2(dist2).squeeze())
        # masked_node = F.relu(self.conv1(masked_node, torch.zeros([2,1], dtype=edge_index.dtype, device= edge_index.device)))
        # s1 = self.score(x, masked_node)
        
        # x_j = torch.index_select(x, 0, total_edge_index[0])
        # x_i = torch.index_select(x, 0, total_edge_index[1])
        # s1 = x_i-x_j
        # 2 layer
        x = self.conv2(x, edge_index)

        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])

        dist3 = x_j-x_i
        o3 = F.relu(self.score3(dist3).squeeze())
        score_loss = torch.matmul(dist1, self.score1.weight.squeeze()).mean()+torch.matmul(dist2, self.score2.weight.squeeze()).mean() +torch.matmul(dist3, self.score3.weight.squeeze()).mean()
  
        # out = F.relu(self.sum(torch.cat(o1,o2,o3),0).squeeze())

        # return torch.einsum("ef,ef->e", x_i, x_j)
        return o3, score_loss


        # return  F.log_softmax(x, dim=1)

class Net_inner(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_inner, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        # self.score = GATScore(Linear(args.hidden*2, 1))
        self.score = Linear(dataset.num_classes, 1)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, pos_edge_index, neg_edge_index, edge_index):
        x,  masked_nodes = data.x, data.masked_nodes
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # x_j = torch.index_select(x, 0, total_edge_index[0])
        # x_i = torch.index_select(x, 0, total_edge_index[1])
        # s0 = F.relu(self.score(x_i-x_j).squeeze())
        # _, net_0 = torch.topk(s0, 10 ,-1 )

        x = F.relu(self.conv1(x, edge_index))

        # masked_node = F.relu(self.conv1(masked_node, torch.zeros([2,1], dtype=edge_index.dtype, device= edge_index.device)))
        # s1 = self.score(x, masked_node)
        
        # x_j = torch.index_select(x, 0, total_edge_index[0])
        # x_i = torch.index_select(x, 0, total_edge_index[1])
        # s1 = x_i-x_j
        # 2 layer5
        x = self.conv2(x, edge_index)

        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])

        # dist = x_j-x_i
        # out = F.relu(self.score(dist).squeeze())
        # score_loss = torch.matmul(dist, self.score.weight.squeeze()).mean()
  

        return torch.einsum("ef,ef->e", x_i, x_j), torch.zeros(total_edge_index.size())
        # return out, score_loss


        # return  F.log_softmax(x, dim=1)

dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
# dataset = dataset.shuffle()
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping)
