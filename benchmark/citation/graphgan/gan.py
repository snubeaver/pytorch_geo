import numpy as np
import torch.nn as nn
import random
from torch.optim import Adam
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import argparse
import pdb
from sklearn.metrics import roc_auc_score, f1_score
import multiprocessing
from torch.distributions import Categorical
import itertools
from torch.optim.lr_scheduler import StepLR

motif_size =3
# max_value = 1000
gen_interval= n_epochs_gen = 3
dis_interval= n_epochs_dis = 3
n_sample = 3
batch_size_gen = 64
batch_size_dis =64  
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(torch.nn.Module):
    def __init__(self, dataset):
        super(Generator, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index)) # LAYER 1
        z = self.conv2(x, edge_index)  # LAYER 2 

        return z



class Discriminator(torch.nn.Module):
    def __init__(self, dataset):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index)) # LAYER 1
        z = self.conv2(x, edge_index)  # LAYER 2 

        return z

lr = 0.001


def train_gan(dataset, data, writer):
    discriminator = Discriminator(dataset)
    discriminator.to(device).reset_parameters()
    generator = discriminator
    # generator = Generator(dataset)
    # generator.to(device).reset_parameters()
    optimizer_d = Adam(discriminator.parameters(), lr=lr)
    optimizer_g = Adam(generator.parameters(), lr=lr)
    id2motifs = build_motifs(data)
    print("start training...")

    # for epoch in range(500):
    #     print("epoch %d" % epoch)
    #     loss_d = train_d(discriminator, optimizer_d, data, id2motifs, generator, writer)
    #     writer.add_scalar('pre/Discriminator loss', loss_d, epoch)

    scheduler = StepLR(optimizer_g, step_size=1000, gamma=0.5)

    for epoch in range(3000):
        print("epoch %d" % epoch)
        # for i in range(1000):
        loss_d = train_d(discriminator, optimizer_d, data, id2motifs, generator, writer)
        # writer.add_scalar('pre/Discriminator loss', loss_d, i)

        loss_g = train_g(generator, optimizer_g, data, id2motifs, discriminator, writer)
        scheduler.step()

        writer.add_scalar('pre/Generator loss', loss_g, epoch)
        writer.add_scalar('pre/Discriminator loss', loss_d, epoch)

        if(epoch%1==0):
            # auc = evaluate(generator, data, data.test_pos_edge_index, data.test_neg_edge_index)
            auc, acc = evaluate(generator, data, data.train_pos_edge_index, data.train_neg_edge_index)
            writer.add_scalar('pre/auc score', auc, epoch)
            writer.add_scalar('pre/acc', acc, epoch)


    print("training completes")


# def evaluate(model, )


def train_d(model, optimizer, data, id2motifs, generator, writer):
    motifs = []
    labels = []
    epoch=0
    losses=[]
    for d_epoch in range(n_epochs_gen):

        # generate new subsets for the discriminator for every dis_interval iterations
        if d_epoch % dis_interval == 0:
            motifs, labels = prepare_data_for_d(data, id2motifs, generator)

        # training30
        train_size = len(motifs)
        motif = motifs
        label = labels
        label = torch.tensor(label).to(device)
        z = model(data.x, data.total_edge_index)

        motif = [list(i) for i in motif]
        motif  = torch.tensor(motif)
        # 
        score = torch.sum((torch.prod(z[motif[:,[0,1]]], axis=1)+torch.prod(z[motif[:,[1,2]]], axis=1)+torch.prod(z[motif[:,[0,2]]], axis=1)), axis=1)
        # pd = torch.prod(z[motif], axis=1)
        # score = torch.sum( pd, axis=1)
        p = torch.sigmoid(score)
        loss = -(torch.sum(label * p + (1 - label) * (1 - p)))
        total_edge_index = torch.cat([data.train_pos_edge_index, data.train_neg_edge_index], dim=-1)
        x_j = torch.index_select(z, 0, total_edge_index[0])
        x_i = torch.index_select(z, 0, total_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)
        link_labels = get_link_labels(data.train_pos_edge_index, data.train_neg_edge_index)
        # loss += F.binary_cross_entropy_with_logits(link_logits, link_labels)
        # print(loss.item())
        losses.append(loss)
        loss.backward()
        optimizer.step()
    losses = torch.tensor(losses)
    return losses.mean().item()
        

def reward_d(model, data, motifs):
    z = model(data.x, data.total_edge_index)
    motif = torch.tensor([list(i) for i in motifs])
    score = torch.sum((torch.prod(z[motif[:,[0,1]]], axis=1)+torch.prod(z[motif[:,[1,2]]], axis=1)+torch.prod(z[motif[:,[0,2]]], axis=1)), axis=1)
    p = torch.sigmoid(score)
    reward = 1-p
    return reward



def train_g(model, optimizer, data,  id2motifs, discriminator,writer):
    motifs = []
    epoch=0
    losses=[]
    for g_epoch in range(n_epochs_gen):

        # generate new subsets for the generator for every gen_interval iterations
        if g_epoch % gen_interval == 0:
            motifs, rewards = prepare_data_for_g(data, id2motifs, model,  discriminator)

        # training
        train_size = len(motifs)
        start_list = list(range(0, train_size, batch_size_gen))
        np.random.shuffle(start_list)
        motif = torch.tensor([list(i) for i in motifs])
        reward = rewards
        reward = torch.tensor(reward).to(device)

        z = model(data.x, data.train_edge_index)

        score = torch.sum((torch.prod(z[motif[:,[0,1]]], axis=1)+torch.prod(z[motif[:,[1,2]]], axis=1)+torch.prod(z[motif[:,[0,2]]], axis=1)), axis=1)
        # p = 1 - torch.exp(-score)
        # p = torch.clamp(p, 1e-5, 1)
        p = torch.sigmoid(score)
        loss = -torch.mean(p*reward)    

        total_edge_index = torch.cat([data.train_pos_edge_index, data.train_neg_edge_index], dim=-1)
        x_j = torch.index_select(z, 0, total_edge_index[0])
        x_i = torch.index_select(z, 0, total_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)
        link_labels = get_link_labels(data.train_pos_edge_index, data.train_neg_edge_index)
        # loss += F.binary_cross_entropy_with_logits(link_logits, link_labels)



        losses.append(loss)
        loss.backward()
        optimizer.step()
    losses = torch.tensor(losses)
    return losses.mean().item()



def prepare_data_for_d(data, id2motifs, generator):
    """generate positive and negative samples for the discriminator"""
    motifs = []
    labels = []
    g_s_args = []
    poss = []
    negs = []
    for i in range(data.x.size(0)):
        if np.random.rand() < 0.5:
            pos = random.sample(id2motifs[i], min(len(id2motifs[i]), n_sample))
            poss.append(pos)
            g_s_args.append((i, len(pos), True))


    z = generator(data.x, data.total_edge_index)
    negs, _ = sampling(g_s_args, z, data)

    for pos in poss:
        if len(pos) != 0:
            motifs.extend(pos)
            labels.extend([1] * len(pos))
    motifs+=negs
    labels.extend([0] * len(negs))
    motifs, labels = shuffle(motifs, labels)
    return motifs, labels

def shuffle(*args):
    idx = list(range(len(args[0])))
    random.shuffle(idx)
    results = []
    for array in args:
        results.append([array[i] for i in idx])
    return tuple(results)
#  data.train_edge_index, data.train_masked_nodes

def prepare_data_for_g(data, id2motifs, generator,  discriminator):
    """sample subsets for the generator"""

    paths = []
    g_s_args = []
    for i in data.train_nodes:
        if np.random.rand() < 0.5:
            g_s_args.append((i, n_sample, False))

    z = generator(data.x, data.train_edge_index)

    motifs, paths = sampling(g_s_args, z, data)
      
    rewards = reward_d(discriminator, data, motifs)
    
    
    motifs, reward = shuffle(motifs, rewards)
    return motifs, reward



def build_motifs(data):
    x = data.x
    id2nid = build_nid(data)
    motifs = set((node, ) for node in data.train_nodes)
    id2motifs = [[] for i in range(x.size(0))]
    num =0
    for i in  range(x.size(0)):
        comb = list(itertools.combinations(id2nid[i], r=2))
        if(len(comb)>0):
            motifs = set(tuple(sorted(list(motif) + [i])) for motif in comb)
            num +=len(motifs)
            for k in motifs:
                id2motifs[i].append(k) 
    # pdb.set_trace()
    print('totally %d motifs' % num)
    data.id2motifs = id2motifs
    return id2motifs

def build_nid(data):
    row, col = data.total_edge_index
    col = col.tolist()
    id2nid= []
    for i in range(data.x.size(0)):
        id2nid.append([])
    key =0
    temp=[]
    for i, item in enumerate(col):
        if(row[i]==key):
            temp.append(item)
        else:
            id2nid[row[i]]=temp
            temp=[]
            key=row[i]
    id2nid = [set(nodes) for nodes in id2nid]
    data.id2nid= id2nid
    return id2nid





def evaluate(model, data, pos_edge_index, neg_edge_index ):
    model.eval()
    z = model(data.x, data.total_edge_index)
    pos_edge_index, neg_edge_index = data.val_pos_edge_index, data.val_neg_edge_index
    total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    x_j = torch.index_select(z, 0, total_edge_index[0])
    x_i = torch.index_select(z, 0, total_edge_index[1])
    score = torch.einsum("ef,ef->e", x_i, x_j)
    link_probs = torch.sigmoid(score)
    pred = link_probs>0.5
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    acc = pred.eq(link_labels).sum().item() / link_probs.size(0)


    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    # pdb.set_trace()
    return roc_auc_score(link_labels, link_probs>0.5), acc


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def sampling(pl, z, data):  # for multiprocessing, pass multiple args in one tuple
    motifs = []
    paths = []
    for k in pl:
        root, n_sample, only_neg = k
        if(root not in data.train_nodes): continue
        motif = [root]
        v1, v2, v3 = g_v(motif, z, data)
        for i in range(n_sample):
            if(np.random.rand() < 0.5): continue
            motif = [root]
            if (i==1):    
                motif.append(v1)
                motif.append(v2)
                motif = tuple(sorted(motif))
            elif(i==2):    
                motif.append(v1)
                motif.append(v3)
                motif = tuple(sorted(motif))
            elif(i==3):
                motif.append(v2)
                motif.append(v3)
                motif = tuple(sorted(motif))
            if(len(motif)<motif_size):
                continue
            motifs.append(motif)
    return motifs, paths
def g_v(roots, z, data):
    g_v_v = z[roots[0]]
    all_node =list(range(z.size(0)))
    all_node.pop(roots[0])
    row = torch.tensor(all_node).to(device)

    x_j = torch.index_select(z, 0, row)
    x_i = g_v_v.repeat(z.size(0)-1,1)
    one_hop = torch.einsum("ef,ef->e", x_i, x_j)
    rel_prob = torch.softmax((1-torch.exp(-one_hop)), -1)
    v1 = torch.multinomial(rel_prob,1).item()
    rel_prob[v1] = 0
    v2 = torch.multinomial(rel_prob,1).item()
    rel_prob[v2] = 0
    v3 = torch.multinomial(rel_prob,1).item()
    

    # prob_dist = torch.distributions.Categorical(rel_prob)
    return v1, v2, v3