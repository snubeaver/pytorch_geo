#!/bin/sh

echo "Cora"
echo "===="
echo "Cold start"
CUDA_VISIBLE_DEVICES=1 python cold_start.py --dataset=Cora
