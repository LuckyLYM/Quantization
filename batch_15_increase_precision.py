#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# nslice form 1 to 7
gpu_list=[2,3,4,5,6,7]
#gpu_list2=[1,2,3,4,5,6,7,1,2,3,4,5,6,7]
#layer_list=[1,2,4,6,8,10]
# python dq.py --quantization=data --arch=DARTS --dataset=CIFAR100 --slice=1 --layers=1 --gpu=4

def run(index,gpu):  


    dataset_list=['SVHN','CIFAR10','CIFAR100']
    dataset=dataset_list[index%3]

    if index<3:
        cmdLine = "python dq.py --quantization=none --arch=DARTS --epochs=100 --dataset=%s --gpu=%d --layers=6"%(dataset,gpu)
    else:
        cmdLine = "python dq.py --quantization=data --arch=DARTS --data_schedule=batch_increase --epochs=100 --dataset=%s --gpu=%d --layers=6"%(dataset,gpu)

    print(cmdLine)
    os.system(cmdLine)


def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)

    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(run,(index,gpu))


    pool.close()
    pool.join()

    
if __name__ == "__main__":

    batchRun(6)

    
