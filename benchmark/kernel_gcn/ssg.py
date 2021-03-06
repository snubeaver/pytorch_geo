from math import ceil
from torch.nn import Linear
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv, JumpingKnowledge
from model_graph import  dense_diff_pool,dense_ssgpool_gumbel, get_spectral_loss_mini_eigen
import pdb


def l2norm(X, eps=1e-10):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm+eps)
    return X

class GNN_Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_Block, self).__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels + hidden_channels, out_channels)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        # self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        out = self.lin(torch.cat((x1, x2), -1))

        return out


class SSGPool(nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.1, lambda_=0.1):
        super(SSGPool, self).__init__()
        self.hidden = hidden

        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.embed_block1 = GNN_Block(dataset.num_features, hidden, hidden) #DenseGCNConv(word_dim, hidden) #
        self.pool_block1 = GNN_Block(dataset.num_features, hidden, num_nodes) #DenseGCNConv(hidden, 20) # 
        self.lambda_=lambda_
        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(GNN_Block(hidden, hidden, hidden))
            self.pool_blocks.append(GNN_Block(hidden, hidden, num_nodes))
        self.embed_final = GNN_Block(hidden, hidden, hidden) #DenseGCNConv(hidden, hidden) #
        self.jump = JumpingKnowledge(mode='cat')
        
        self.lin1 = nn.Linear(hidden * (num_layers+1)  , hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        x, adj, mask = data.x, data.adj, data.mask

        spec_losses = 0.
        spec_losses_hard = 0.

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        B, N, _ = adj.size()




        s_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_soft_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_inv_soft_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()

        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [torch.sum(x, 1) / (mask.sum(-1, keepdims=True).to(x.dtype) + 1e-10)]

        diag_ele = torch.sum(adj, -1)
        Diag = torch.diag_embed(diag_ele)
        Lapl = Diag - adj
        Lapl_ori = Lapl

        x, adj, L_next, L_next_soft, s, s_soft, s_inv, s_inv_soft = dense_ssgpool_gumbel(x, adj, s, Lapl, Lapl, mask,
                                                                                         is_training=self.training)

        s_final = torch.bmm(s_final, s)
        s_soft_final = torch.bmm(s_soft_final, s_soft)
        s_inv_final = torch.bmm(s_inv, s_inv_final)
        s_inv_soft_final = torch.bmm(s_inv_soft, s_inv_soft_final)

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj, add_loop=True)
            x = F.relu(embed_block(x, adj, add_loop=True))
            xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks):
                x, adj, L_next, L_next_soft, s, s_soft, s_inv, s_inv_soft = dense_ssgpool_gumbel(x, adj, s, L_next,
                                                                                                 L_next_soft,
                                                                                                 is_training=self.training)
                s_final = torch.bmm(s_final, s)
                s_soft_final = torch.bmm(s_soft_final, s_soft)
                s_inv_final = torch.bmm(s_inv, s_inv_final)
                s_inv_soft_final = torch.bmm(s_inv_soft, s_inv_soft_final)

        x = self.embed_final(x, adj, add_loop=True)
        xs.append(x.mean(dim=1))

        # feature_out = x.mean(dim=1)

        if self.lambda_ != 0.0 and x.requires_grad ==True:
            fv = data['fv']
            fv = fv.unsqueeze(0) if fv.dim() == 2 else fv
            spec_loss, spec_loss_soft = get_spectral_loss_mini_eigen(fv, s_final, s_inv_final, s_soft_final, s_inv_soft_final, Lapl_ori, mask)
            spec_losses += spec_loss
            #spec_losses += torch.Tensor([0.])
        else:
            spec_losses += torch.Tensor([0.])


        spec_losses = spec_losses.mean()

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), self.lambda_ * (spec_losses)







    def __repr__(self):
        return self.__class__.__name__