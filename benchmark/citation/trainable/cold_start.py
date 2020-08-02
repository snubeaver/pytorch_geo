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
        num_nodes = dataset[0].num_nodes
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.p = torch.nn.Parameter(torch.randn(int(num_nodes*0.1), num_nodes), requires_grad=True)
        # test_mask x all_node * all_node x feature
        # self.score = Linear(dataset.num_classes, 1)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, pos_edge_index, neg_edge_index):
        x, edge_index, masked_nodes = data.x, data.train_edge_index, data.masked_nodes        
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # mask_feature = x[masked_nodes]
        x_j = torch.index_select(x, 0,  torch.tensor(masked_nodes).cuda())
        x_i = torch.index_select(x, 0,  torch.tensor([e for e in range(x.size(0)) if e not in masked_nodes]).cuda())

        # dist = x_j-x_i
        # out = F.relu(self.score(dist).squeeze())
        # score_loss = torch.matmul(dist, self.score.weight.squeeze()).mean()
  

        first = torch.einsum("ef,bf->eb", x_j, x_i)
        pdb.set_trace()     
    

        



        x = F.relu(self.conv1(x, edge_index))

        x = self.conv2(x, edge_index)

        # x_mask = torch.index_select(x, 0, masked_nodes)
        x[masked_nodes] = torch.matmul(p, x)

        return  F.log_softmax(x, dim=1)
        # dist = x_j-x_i
        # out = F.relu(self.score(dist).squeeze())
        # score_loss = torch.matmul(dist, self.score.weight.squeeze()).mean()
  

        # return torch.einsum("ef,ef->e", x_i, x_j)
        # return out, score_loss

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

    def forward(self, data, pos_edge_index, neg_edge_index):
        x, edge_index, masked_nodes = data.x, data.train_edge_index, data.masked_nodes
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

run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping)
