#!/bin/sh

echo "Cora"
echo "===="
echo "Cold start"
CUDA_VISIBLE_DEVICES=1 python ssl_start.py --dataset=Cora --random_splits=True

