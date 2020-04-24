import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot, zeros
import pdb

class GCNConv_(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha, improved=False,  cached=True,
                 bias=True,normalize=True, **kwargs):
        super(GCNConv_, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.alpha =alpha
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        # edge_index는 [2, num_edge] 형태, 순서대로 row, column index가 됨
        # edge_weight는 num_edges만큼

        fill_value = 1 if not improved else 2
        # self loop 1 더해줌
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        # degree sum of edge weights 
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # result still in COO form 
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        # 처음인데 size 틀리면 에러
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            # random graph generation
            device = edge_index.device
            random_index = torch.randint(0,x.size(0)-1,int(size_edge/3)).to(device)
            
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            random_index, random_norm = self.norm(random_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        gcn_origin = (1-self.alpha)*self.propagate(edge_index, x=x, norm=norm)
        gcn_high = self.alpha*self.propagate(random_index, x=x, norm=random_norm)

        return gcn_origin+gcn_high

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
