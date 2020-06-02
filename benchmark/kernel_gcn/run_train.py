from argparse import ArgumentParser
import subprocess
import os

DATA = "['NCI1', 'NCI109','Mutagenicity', ]"


def run_train():
    subprocess.call(['python', 'main_ssg.py', '--datasets', 'Mutagenicity', '--ratio', '0.1'])


if __name__ == "__main__":
    parser = ArgumentParser(description='training code')
    parser.add_argument('-gpu', default="0", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_train()