import numpy as np
import torch.nn as nn
import random
from torch.optim import Adam
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import argparse
import pdb
from sklearn.metrics import roc_auc_score
import multiprocessing

motif_size =3
max_value = 1000
gen_interval= n_epochs_gen = 3
dis_interval= n_epochs_dis = 3
n_sample = 5
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
    generator = Generator(dataset)
    generator.to(device).reset_parameters()
    optimizer_d = Adam(discriminator.parameters(), lr=lr)
    optimizer_g = Adam(generator.parameters(), lr=lr)
    id2motifs = build_motifs(data)
    print("start training...")
    for epoch in range(3000):
        print("epoch %d" % epoch)
        loss_d = train_d(discriminator, optimizer_d, data, id2motifs, generator, writer)
        loss_g = train_g(generator, optimizer_g, data, id2motifs, discriminator, writer)
        
        writer.add_scalar('pre/Generator loss', loss_g, epoch)
        writer.add_scalar('pre/Discriminator loss', loss_d, epoch)

        if(epoch%10==0):
            auc = evaluate(generator, data, data.test_pos_edge_index, data.test_neg_edge_index)
            writer.add_scalar('pre/auc score', auc, epoch)


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
        '''
        start_list = list(range(0, train_size, batch_size_dis))
        np.random.shuffle(start_list)
        for start in start_list:

            
            end = start + batch_size_dis
            motif = motifs[start:end]
            label = labels[start:end]
            # if(np.array(motif).shape[0]<8):
            #     break
            label = torch.tensor(label).to(device)
            z = model(data.x, data.train_edge_index)
            pd = torch.prod(z[motif], axis=1)
            score = torch.sum( pd, axis=1)
            p = 1 - torch.exp(-score)
            p = torch.clamp(p, min=1e-5, max=1)
            
            loss = -(torch.sum(label * p + (1 - label) * (1 - p)))
            losses.append(loss)
            loss.backward()
            optimizer.step()
        '''
        motif = motifs
        label = labels
        label = torch.tensor(label).to(device)
        z = model(data.x, data.train_edge_index)
        # pdb.set_trace()
        pdb.set_trace()
        pd = torch.prod(z[motif], axis=1)
        score = torch.sum( pd, axis=1)
        p = 1 - torch.exp(-score)
        p = torch.clamp(p, min=1e-5, max=1)
        loss = -(torch.sum(label * p + (1 - label) * (1 - p)))
        losses.append(loss)
        loss.backward()
        optimizer.step()
    losses = torch.tensor(losses)
    return losses.mean().item()
        

def reward_d(model, data, motif):
    z = model(data.x, data.train_edge_index)
    score = torch.sum(torch.prod(z[motif], axis=1), axis=1)
    p = 1 - torch.exp(-score)
    # p = torch.sigmoid(score)
    p = torch.clamp(p, 1e-5, 1)
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
        '''
        for start in start_list:
            end = start + batch_size_gen
            motif = motifs[start:end]
            reward = rewards[start:end]
            reward = torch.tensor(reward).to(device)

            z = model(data.x, data.train_edge_index)
            score = torch.sum(torch.prod(z[motif], axis=1), axis=1)
            # p = 1 - torch.exp(-score)
            # p = torch.clamp(p, 1e-5, 1)
            p = torch.sigmoid(score)
            loss = -torch.mean(p*reward)    
            losses.append(loss)
            loss.backward()
            optimizer.step()
        '''
        motif = motifs
        reward = rewards
        reward = torch.tensor(reward).to(device)

        z = model(data.x, data.train_edge_index)
        score = torch.sum(torch.prod(z[motif], axis=1), axis=1)
        # p = 1 - torch.exp(-score)
        # p = torch.clamp(p, 1e-5, 1)
        p = torch.sigmoid(score)
        loss = -torch.mean(p*reward)    
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
        if np.random.rand() < 1:
            pos = random.sample(id2motifs[i], min(len(id2motifs[i]), n_sample))
            poss.append(pos)
            g_s_args.append((i, len(pos), True))


    z = generator(data.x, data.total_edge_index)
    # row, col = data.total_edge_index


    # x_j = torch.index_select(z, 0, row)
    # x_i = torch.index_select(z, 0, col)
    # one_hop = torch.einsum("ef,ef->ef", x_i, x_j)

    negs, _ = sampling(g_s_args, z, data)

    # negs =[]
    # for i in range(data.x.size(0)):
    #     neg=[]
    #     if(len(poss[i])>0):
    #         ps= torch.tensor(poss[i][0]).to(device)
    #         # pdb.set_trace()
    #         x_j = torch.index_select(one_hop, 0, ps)
    #         x_i = torch.index_select(one_hop, 0, ps)
    #         two_hop = torch.einsum("ef,ef->e", x_j, x_i)
    #         __, target = torch.topk(two_hop, len(poss[i]))
    #         for k in range(len(poss[i])):
    #             neg.append((i, row[target[k]].item(), col[target[k]].item()))
    #     negs.append(neg)

    
    for pos, neg in zip(poss, negs):
        if len(pos) != 0 and neg is not None:
            motifs.extend(pos)
            labels.extend([1] * len(pos))
            motifs+=neg
            labels.extend([0] * len(neg))
    motifs, labels = shuffle(motifs, labels)
    pdb.set_trace()
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
    for i in range(data.x.size(0)):
        if np.random.rand() < 1:
            g_s_args.append((i, n_sample, False))

    z = generator(data.x, data.total_edge_index)

    motifs, paths = sampling(g_s_args, z, data)
    # pdb.set_trace()
    # motifs = [j for i in motifs for j in i]
    
    '''
    row, col = data.total_edge_index

    x_j = torch.index_select(z, 0, row)
    x_i = torch.index_select(z, 0, col)
    one_hop = torch.einsum("ef,ef->ef", x_i, x_j)

    '''

    '''
    motifs=[]
    for neg in negs:
        motifs.extend(neg)
    '''
    rewards = []
   
    rewards.append(reward_d(discriminator, data, motifs).tolist())
    rewards = np.concatenate(rewards)
    
    motifs, reward = shuffle(motifs, rewards)
    return motifs, reward



def build_motifs(data):
    x = data.x
    id2nid = build_nid(data)
    motifs = set((node, ) for node in range(x.size(0)))
    for i in range(motif_size - 1):
        print('getting motifs with size of %d' % (i + 2))
        motifs = get_motifs_with_one_more_node(motifs, id2nid)
        print('totally %d motifs' % len(motifs))
    # pdb.set_trace()
    id2motifs = [[] for i in range(x.size(0))]
    for motif in motifs:
        for nid in motif:
            id2motifs[nid].append(motif)
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


def get_motifs_with_one_more_node(motifs, id2nid):
    motifs_next = set()
    for motif in motifs:
        nei = id2nid[motif[0]] - set(motif)
        for node in motif[1:]:
            # nei = nei & id2nid[node]
            nei.update(id2nid[node])
        for node in nei:
            motifs_next.add(tuple(sorted(list(motif) + [node])))
    return motifs_next
import multiprocessing




def evaluate(model, data, pos_edge_index, neg_edge_index ):
    model.eval()
    z = model(data.x, data.total_edge_index)
    total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    x_j = torch.index_select(z, 0, data.total_edge_index[0])
    x_i = torch.index_select(z, 0, data.total_edge_index[1])
    one_hop = torch.einsum("ef,ef->ef", x_i, x_j)

    x_j = torch.index_select(one_hop, 0, total_edge_index[0])
    x_i = torch.index_select(one_hop, 0, total_edge_index[1])
    score = torch.einsum("ef,ef->e", x_j, x_i)
    link_probs = torch.sigmoid(score)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    auc = (roc_auc_score(link_labels, link_probs))
    return auc


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels
# def sampling(p, z, data):  # for multiprocessing, pass multiple args in one tuple

#     motifs, paths = zip(*multiprocessing.Pool(16).map(g_s, [p, z, data]))
#     return motifs, paths
def sampling(pl, z, data):  # for multiprocessing, pass multiple args in one tuple
    motifs = []
    paths = []
    for k in range(len(pl)):
        root, n_sample, only_neg = pl[k]
        
        for i in range(2 * n_sample):
            if len(motifs) >= n_sample:
                break
            motif = [root]
            path = [root]
            for j in range(1, motif_size):
                v, p = g_v(motif, z, data)
                if v is None:
                    break
                motif.append(v)
                path.extend(p)
            if len(set(motif)) < motif_size:
                continue
            motif = tuple(sorted(motif))
            # if only_neg:
            #     continue
            motifs.append(motif)
            paths.append(path)
    return motifs, paths
def g_v(roots, z, data):
    g_v_v = z[roots[0]]
    for nid in roots[1:]:
        g_v_v *= z[nid]
    current_node = roots[-1]
    previous_nodes = set()
    path = []
    is_root = True
    while True:
        if is_root:
            # node_neighbor = list({neighbor for root in roots for neighbor in data.id2nid[root]})
            node_neighbor = [x for x in list(range(data.x.size(0))) if x not in roots]
        else:
            # node_neighbor = [x for x in list(range(data.x.size(0))) if x not in roots]
            node_neighbor = list(data.id2nid[current_node])
        if len(node_neighbor) == 0:  # the root node has no neighbor
            return None, None
        if is_root:
            tmp_g = g_v_v
        else:
            tmp_g = g_v_v * z[current_node]
        relevance_probability = torch.sum(z[node_neighbor] * tmp_g, axis=1)
        relevance_probability = torch.softmax(1-torch.exp(-relevance_probability), dim=-1)
        s = torch.sum(relevance_probability)
        relevance_probability = relevance_probability.tolist()
        target = (random.random() * s).item()
        for i, wi in enumerate(relevance_probability):
            if target < wi:
                next_node = node_neighbor[i]
                break
            else:
                target -= wi
                next_node = node_neighbor[i]

        if next_node in previous_nodes or len(path)>2:  # terminating condition
            break
        previous_nodes.add(current_node)
        current_node = next_node
        path.append(current_node)
        is_root = False
    return current_node, path
