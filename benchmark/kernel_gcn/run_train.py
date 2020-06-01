from argparse import ArgumentParser
import subprocess
import os

N_SPLIT = 0
EPOCH = 19
DATA = 'MUTAG'
path = '/data/project/rw/kloud/graph_benchmark/results/'

#'--sbert', 'bert-base-nli-stsb-mean-tokens',
def run_train():
    subprocess.call(['python', 'main_ssg.py', '>', path+DATA])


if __name__ == "__main__":
    parser = ArgumentParser(description='training code')
    parser.add_argument('-gpu', default="0", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_train()