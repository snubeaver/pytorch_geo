from itertools import product

import argparse
import torch
import torch.nn.functional as F
from gcn_diff import GCNDiff
from gcn_high import GCNHIGH
from gcn import GCN
from datasets import get_planetoid_dataset
from train_eval import random_planetoid_splits, run
from train_eval_diff import run_
import torch_geometric.transforms as T


random_splits = False
layers = [2, 3]
# layers = [4]
hidden = 16
alphas = [1,2,3,4,5]
run_num = 100
epoch_num =300
datasets = ['Cora', 'CiteSeer','PubMed'] 
nets = [
    GCNHIGH,
    GCNDiff,
    GCN,
]


results = []
for dataset_name, Net in product(datasets, nets):
    print("{}\n ========".format(dataset_name))
    dataset = get_planetoid_dataset(dataset_name, True)
    permute_masks = random_planetoid_splits if random_splits else None
    if(Net==GCNDiff):
        print("GCN Diffusion")
        data = dataset[0]
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
                    sparsification_kwargs=dict(method='topk', k=128,
                                                dim=0), exact=True)
        data = gdc(data)
        run_(dataset, data, Net(data, dataset, hidden), run_num, epoch_num, 0.01, 0.0005,
            10, permute_masks)
    elif(Net==GCNHIGH):
        print("GCN Highway")
        for alpha in alphas:
            print("alpha : {}".format(alpha))
            
            run(dataset, Net(dataset,hidden, alpha), run_num, epoch_num, 0.01, 0.0005,
                10, permute_masks)
    else:
        print("GCN")
        run(dataset, Net(dataset, hidden), run_num, epoch_num, 0.01, 0.0005,
            10, permute_masks)