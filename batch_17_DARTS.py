#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# nslice form 1 to 7
gpu_list=[2,3,4,5,6,7]
gpu_list2=[2,3,4,5,6,7,2,3,4,5,6,7]
dataset_list=['CIFAR10','CIFAR100']
bits_list=[3,4,5,6,7,8]
split_list=[1.0,0.75,0.6,0.5,0.43,0.375]


# python dq.py --quantization=data --arch=DARTS --dataset=CIFAR100 --slice=1 --layers=1 --epochs=1 --train_split=0.6 --gpu=1

def run(index,gpu):  

    layers=6
    dataset=dataset_list[index//6]
    nslice=bits_list[index%6]
    split=split_list[index%6]



    cmdLine = "python dq.py --quantization=data --epochs=200 --arch=%s --dataset=%s --gpu=%d --slice=%d --layers=%d --train_split=%f"%(arch,dataset,gpu,nslice,layers,split)

    print(cmdLine)
    os.system(cmdLine)


def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list2):  
        pool.apply_async(run,(index,gpu))

    pool.close()
    pool.join()

    
if __name__ == "__main__":
    

    arch='DARTS'
    batchRun(12)


    
