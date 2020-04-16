#!/bin/sh

echo "Cora"
echo "===="

echo "GCN_HIGH"
python gcn_high.py --dataset=Cora
python gcn_high.py --dataset=Cora --random_splits=True


echo "GCN_DIFFUSION"
python gcn_diff.py --dataset=Cora
python gcn_diff.py --dataset=Cora --random_splits=True




echo "GCN"
python gcn.py --dataset=Cora
python gcn.py --dataset=Cora --random_splits=True



echo "CiteSeer"
echo "========"

echo "GCN"
python gcn.py --dataset=CiteSeer
python gcn.py --dataset=CiteSeer --random_splits=True

echo "GAT"
python gat.py --dataset=CiteSeer
python gat.py --dataset=CiteSeer --random_splits=True


echo "PubMed"
echo "======"

echo "GCN"
python gcn.py --dataset=PubMed
python gcn.py --dataset=PubMed --random_splits=True

echo "GAT"
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits=True

