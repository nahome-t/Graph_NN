#!/bin/bash

#export py=/users/hert5217/anaconda3/envs/MLstuff/bin/python3
# export mpi=/usr/local/shared/openmpi/4.0.0/bin/mpiexec
export py=/mnt/zfsusers/nahomet/anaconda3/envs/GNN/bin/python3

# FCN batch

# 1 node, 6gb per node
addqueue -n 1 -m 6 -s $py main.py --arg1 10
