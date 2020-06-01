from itertools import product
import sys
import argparse
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

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
from ssg import SSGPool, SSGPool_gumbel
from mincut import MincutPool

'''
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("/data/project/rw/kloud/graph_benchmark/results/Diff_worker24_MUTAG.log", "a")
        #/data/project/rw/kloud/graph_benchmark/results/
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

con_logger = Logger()
'''
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()

<<<<<<< HEAD
layers = [1]#, 2, 3, 4]
# layers = [4]
hiddens = [16, 32, 64, 128]
lambdas_ = [0.0, 0.001, 0.01, 0.1, 1]
# hiddens = [16]
datasets = ['MUTAG']#'ENZYMES','DD' ,'IMDB-BINARY', 'PROTEINS', 'REDDIT-BINARY', 'MUTAG']  # , 'COLLAB']
# datasets = [ ]
nets = [
    # GCNWithJK,
    DiffPool,
    # MincutPool,
    # SAGPool,
    # # SSGPool,
=======
layers = [1, 2]
ratios = [0.1, 0.2, 0.4, 0.8]
hiddens = [128]
datasets = ['ENZYMES','MUTAG', 'IMDB-BINARY', 'PROTEINS','REDDIT-BINARY' ]  # , 'COLLAB']
# datasets = ['MUTAG' ]
nets = [
    # GCNWithJK,
    DiffPool,
    MincutPool,
    SAGPool,
    # SSGPool,
>>>>>>> b85669844b6d7eb351ce3c20e0b450588aecc910
    # # GraphSAGEWithJK,
    # # GIN0WithJK,
    # # GINWithJK,
    # # Graclus,
    TopK,
    # 
    EdgePool,
    # GCN,
    # GraphSAGE,
    # GIN0,
    # GIN,
    # GlobalAttentionNet,
    # Set2SetNet,
    # SSGPool,
    # SSGPool_gumbel,
    # DiffPool,
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
    if(Net == DiffPool or Net == SSGPool or Net == SSGPool_gumbel or Net == MincutPool): diff=True
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
<<<<<<< HEAD
    for lambda_, num_layers, hidden in product(lambdas_, layers, hiddens):
        dataset = get_dataset(dataset_name, sparse = not diff)
        # dataset = get_dataset(dataset_name, sparse = True)
        model = Net(dataset, num_layers, hidden, lambda_ = lambda_)
=======
    for num_layers, hidden, ratio in product(layers, hiddens, ratios):
        dataset = get_dataset(dataset_name, sparse = not diff)
        # dataset = get_dataset(dataset_name, sparse = True)
        # dataset[0] = pre_eig(dataset[0])
        model = Net(dataset, num_layers, hidden,ratio)
>>>>>>> b85669844b6d7eb351ce3c20e0b450588aecc910
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
        if acc > best_result[1]: # loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))
