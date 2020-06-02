from math import ceil

import torch
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter


import time
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv,  JumpingKnowledge
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax

EPS = 1e-10

class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, x, adj, mask=None, add_loop=False):

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

def arccosh(x):
    x = x + EPS
    #c0 = torch.log(x)
    ##print(c0)
    #c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    #return c0 + c1
    a = torch.sqrt(x*x-1)
    return torch.log(x+a)




def dense_diff_pool(x, adj, s, mask=None):
    r"""Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns pooled node feature matrix, coarsened adjacency matrix and two
    auxiliary objectives: (1) The link prediction loss

    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|}_F,

    and the entropy regularization

    .. math::
        \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask_ = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask_, s * mask_

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    #identity = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, num_nodes, num_nodes).cuda()
    #link_loss = (adj+identity) - torch.matmul(s, s.transpose(1, 2))
    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    if mask is not None:
        link_loss = link_loss * mask_
        link_loss = link_loss * mask_.transpose(1, 2)

    link_loss = torch.sqrt((link_loss*link_loss).sum(dim=(1, 2)))#torch.norm(link_loss, p=2, dim=-1)
    #link_loss = link_loss.sum(-1) / (mask.sum(-1) + 1e-10)
    #link_loss = link_loss.sum()  # / batch_size # / adj.numel()



    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1)  # .mean()
    if mask is not None:
        ent_loss = ent_loss.sum(-1) / (mask.sum(-1) + 1e-10)
    else:
        ent_loss = ent_loss.mean(-1)
    #ent_loss = ent_loss.sum()

    #spec_loss = get_Spectral_loss(adj, out_adj, s, mask)


    return out, out_adj, link_loss.mean(), ent_loss.mean()

def dense_ssgpool_gumbel(x, adj, s, Lapl, Lapl_soft, mask=None, is_training=False, debug=False):


    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s_in = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    ###s = torch.softmax(s, dim=-1)
    s, s_soft = gumbel_softmax(s_in, is_training=is_training)
    if mask is not None:
        mask_ = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s, s_soft = x * mask_, s * mask_, s_soft * mask_


    #diag_ele = torch.sum(adj, -1)
    #Diag = torch.diag_embed(diag_ele)
    #Lapl = Diag - adj
    L_next = torch.matmul(torch.matmul(s.transpose(1, 2), Lapl), s)
    L_next_soft = torch.matmul(torch.matmul(s_soft.transpose(1, 2), Lapl_soft), s_soft)
    identity = torch.eye(s.size(-1)).unsqueeze(0).expand(batch_size, s.size(-1), s.size(-1)).cuda()
    padding = torch.zeros_like(identity)

    A_next = -torch.where(torch.eq(identity, 1.), padding, L_next)

    '''Get feature matrix of next layer'''

    s_inv_soft = s_soft.transpose(1, 2) / ((s_soft * s_soft).sum(dim=1).unsqueeze(
        -1) + EPS)  # Pseudo-inverse of P (easy inversion theorem, in this case, sum is same with 2-norm)
    s_inv = s.transpose(1, 2) / ((s * s).sum(dim=1).unsqueeze(
        -1) + EPS)  # Pseudo-inverse of P (easy inversion theorem, in this case, sum is same with 2-norm)
    #s_inv_T = P_inv.transpose(1, 2)

    #x_next = torch.bmm(s_inv, x)
    #x_next = torch.bmm(s.transpose(1, 2), x)

    #s_soft_inv = s_soft / (s_soft.sum(1, keepdim=True) + 1e-10)
    x_next = torch.bmm(s_inv_soft, x)
    #x_next = torch.bmm(s_soft.transpose(1, 2), x)
    #identity = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, num_nodes, num_nodes).cuda()
    #link_loss = (adj + identity) - torch.matmul(s, s.transpose(1, 2))

    #return x_next, A_next, L_next_soft, s_soft, s_inv
    return x_next, A_next, L_next, L_next_soft, s, s_soft, s_inv, s_inv_soft


def get_spectral_loss_mini_eigen(x, s, s_inv, s_soft, s_inv_soft, L, mask=None):

    #x = x.unsqueeze(-1)
    x_tilde = torch.bmm(s, torch.bmm(s_inv, x))
    x_tilde_soft = torch.bmm(s_soft, torch.bmm(s_inv_soft, x))
    # x_tilde_soft = torch.bmm(s_soft, x)
    if mask is not None:
        mask_ = mask.unsqueeze(-1)
        x = x * mask_
        x_tilde = x_tilde * mask_
        x_tilde_soft = x_tilde_soft * mask_
        # x_tilde_soft = x_tilde_soft * mask_
    diff = x - x_tilde
    diff_soft = x - x_tilde_soft

    upper = torch.bmm(L, diff)
    upper = torch.bmm(diff.transpose(1, 2), upper)
    upper = torch.sqrt(torch.diagonal(upper, dim1=1, dim2=2) + 1e-10)

    upper_soft = torch.bmm(L, diff_soft)
    upper_soft = torch.bmm(diff_soft.transpose(1, 2), upper_soft)
    upper_soft = torch.sqrt(torch.diagonal(upper_soft, dim1=1, dim2=2) + 1e-10)

    lower = torch.bmm(L, x)
    lower = torch.bmm(x.transpose(1, 2), lower)
    lower = torch.sqrt(torch.diagonal(lower, dim1=1, dim2=2) + 1e-10)

    term_mask = torch.zeros_like(upper)
    upper = torch.where(upper < 0., term_mask, upper)
    upper_soft = torch.where(upper_soft < 0., term_mask, upper_soft)
    lower = torch.where(lower < 0., term_mask, lower)

    spec_loss = upper / (lower + 1e-10)
    spec_loss_soft = upper_soft / (lower + 1e-10)

    spec_loss = spec_loss.mean(-1)
    spec_loss_soft = spec_loss_soft.mean(-1)
    return spec_loss, spec_loss_soft

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, axis, tau=1, eps=1e-10, is_training=False):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """

    # TODO: add noise sampling

    if is_training == True:
        gumbel_noise = sample_gumbel(logits.size(), eps=eps)
        if logits.is_cuda:
            gumbel_noise = gumbel_noise.cuda()
        y = logits #+ gumbel_noise
    else:
        y = logits



    y_soft = F.softmax(y / tau, dim=axis)
    #y_soft = mysoftmax(y / tau, dim=axis, debug=debug)


    #print(y_soft)
    return y_soft


def gumbel_softmax(logits, axis=-1, tau=1, eps=1e-10, is_training=False):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    #logits = logits.permute(0, 2, 3, 1).contiguous()  # multihead to last dim

    y_soft = gumbel_softmax_sample(logits, axis, tau=tau, eps=eps, is_training=is_training)

    #y_soft = torch.where(torch.isnan(y_soft), torch.zeros_like(y_soft), y_soft)

    #print("y_soft")
    #print(y_soft[0][:,0])

    index = y_soft.max(axis, keepdim=True)[1]
    # this bit is based on
    # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5


    y_hard = torch.zeros_like(logits).scatter_(axis, index, 1.0)

    #y_hard = y_hard.zero_().scatter_(axis, index, 1.0)
    # this cool bit of code achieves two things:
    # - makes the output value exactly one-hot (since we add then
    #   subtract y_soft value)
    # - makes the gradient equal to y_soft gradient (since we strip
    #   all other gradients)
    y_hard = y_hard - y_soft.detach() + y_soft

    #if debug == True:

    return y_hard, y_soft  #.permute(0, 3, 1, 2)
