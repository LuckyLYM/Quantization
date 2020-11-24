#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# nslice form 1 to 7
gpu_list=[5,6,7]
gpu_list2=[4,5,6,7,4,5,6,7]
#layer_list=[1,2,4,6,8,10]


def run(index,gpu,layer_list):  

    nslice=gpu
    if len(layer_list)==1 or index<len(gpu_list):
        layers=layer_list[0]
    else:
        layers=layer_list[1]

    # python dq.py --quantization=data --arch=DARTS --dataset=CIFAR10 --gpu=0 --slice=1 --layers=16 --epochs=3
    cmdLine = "python dq.py --quantization=data --arch=%s --dataset=%s --gpu=%d --epochs=200 --slice=%d --layers=%d"%(arch,dataset,gpu,nslice,layers)

    print(cmdLine)
    os.system(cmdLine)


def batchRun(nprocess,layer_list):
    pool = multiprocessing.Pool(processes = nprocess)
    if len(layer_list)==1:
        for index,gpu in enumerate(gpu_list):  
            pool.apply_async(run,(index,gpu,layer_list))
    else:
        for index,gpu in enumerate(gpu_list2):  
            pool.apply_async(run,(index,gpu,layer_list))

    pool.close()
    pool.join()

    
if __name__ == "__main__":
    

    arch='DARTS'

    dataset_list=['CIFAR10']
    for d in dataset_list:
        dataset=d
        '''
        batchRun(8,[1,2])
        batchRun(8,[4,6])
        batchRun(8,[8,10])
        '''
        batchRun(3,[16])

    
