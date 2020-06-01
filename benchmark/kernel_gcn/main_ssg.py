from itertools import product

import argparse
from datasets import get_dataset
from train_eval_ssg import cross_validation_with_val_set

from gcn import GCN, GCNWithJK
from graph_sage import GraphSAGE, GraphSAGEWithJK
from gin import GIN0, GIN0WithJK, GIN, GINWithJK
from graclus import Graclus
from top_k import TopK
from sag_pool import SAGPool
from diff_pool import DiffPool
from edge_pool import EdgePool
from global_attention import GlobalAttentionNet
from set2set import Set2SetNet
from sort_pool import SortPool
from ssg import SSGPool
from mincut import MincutPool

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()

layers = [1, 2]
# layers = [4]
# hiddens = [16, 32, 64, 128]
lambdas_ = [0.00]
# lambdas_ = [0.05, 0.1, 0.5]
ratios = [0.1, 0.2, 0.4, 0.8]
hiddens = [128]
datasets = ['IMDB-BINARY', 'PROTEINS','ENZYMES', 'REDDIT-BINARY' ]  # , 'COLLAB']
# datasets = ['MUTAG' ]
nets = [
    # GCNWithJK,
    # DiffPool,
    # MincutPool,
    # SAGPool,
    SSGPool,
    # # GraphSAGEWithJK,
    # # GIN0WithJK,
    # # GINWithJK,
    # # Graclus,
    # TopK,
    # 
    # EdgePool,
    # GCN,
    # GraphSAGE,
    # GIN0,
    # GIN,
    # GlobalAttentionNet,
    # Set2SetNet,
    # SortPool,
]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))



results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    diff = False
    if(Net == DiffPool or Net == SSGPool or Net == MincutPool): diff=True
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for num_layers, hidden, ratio,  lambda_ in product(layers, hiddens, ratios, lambdas_):
        dataset = get_dataset(dataset_name, sparse = not diff)
        # dataset = get_dataset(dataset_name, sparse = True)
        # dataset[0] = pre_eig(dataset[0])
        model = Net(dataset, num_layers, hidden,ratio, lambda_)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None,
            diff=diff,
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))