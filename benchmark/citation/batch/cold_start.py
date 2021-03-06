import argparse
import torch
import torch.nn.functional as F
from gat_conv import GATConv
from torch.nn import Linear
from datasets import get_planetoid_dataset
from train_eval_cs import run
import pdb
from torch_geometric.nn import GCNConv
from torch.nn import Parameter

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
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.hidden)

        self.conv1_ssl = GCNConv(dataset.num_features, args.hidden)
        self.conv2_ssl = GCNConv(args.hidden, dataset.num_classes)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv1_ssl.reset_parameters()
        self.conv2_ssl.reset_parameters()

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, data, pos_edge_index, neg_edge_index, edge_index):
        x,  masked_nodes = data.x,  data.masked_nodes
        x = F.relu(self.conv1(x, edge_index)) # LAYER 1
        z = self.conv2(x, edge_index)  # LAYER 2 
        
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        total_pred = torch.cat([pos_pred, neg_pred], dim=-1)
        r, c = total_edge_index[0][total_pred>0.5], total_edge_index[1][total_pred>0.5]
        new_index = torch.stack((torch.cat([r,c], dim= -1),(torch.cat([c,r], dim= -1))), dim=0 )
        added_index = torch.cat([edge_index, new_index], dim=-1)
        
        x = data.x
        x = F.relu(self.conv1_ssl(x, added_index)) # LAYER 1
        x = self.conv2_ssl(x, added_index)  # LAYER 2 
        return F.log_softmax(x, dim=1), z,r.size()


class Net_SSL(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_SSL, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, data, pos_edge_index, neg_edge_index, edge_index):
        x,  masked_nodes = data.x,  data.masked_nodes
        # mask_loop = torch.stack((torch.LongTensor(masked_nodes), torch.LongTensor(masked_nodes)), dim=0 )
        mask_loop = torch.stack((torch.LongTensor([0,1]), torch.LongTensor([2,3])), dim=0 )
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)


        x = F.relu(self.conv1(x, edge_index)) # LAYER 1
        x = self.conv2(x, edge_index)  # LAYER 2 
        return F.log_softmax(x, dim=1)


        # return  F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
# dataset = dataset.shuffle()
run(dataset, Net(dataset), Net_SSL(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping)
