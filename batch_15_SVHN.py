#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# 14 index
gpu_list=[1,2,3,4,5,6,7,1,2,3,4,5,6,7]
 


def run(index,gpu):  
    nslice=2*index+1
    # 1,3,...,25
    time.sleep(3*index)

    if nslice==27:
        cmdLine = "python dq.py --arch=%s --dataset=%s --gpu=%d --quantization=none"%(arch,dataset,gpu)
    else:
        cmdLine = "python dq.py --arch=%s --dataset=%s --gpu=%d --slice=%d"%(arch,dataset,gpu,nslice)

    print(cmdLine)
    os.system(cmdLine)


def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(run,(index,gpu))
                      
    pool.close()
    pool.join()

# python dq.py --arch=resnet --dataset=CIFAR100 --gpu=2 --quantization=none

def runNone(index,gpu):  
    dataset='SVHN'
    if index==0:
        arch='nin'
        cmdLine = "python dq.py --arch=%s --dataset=%s --gpu=%d --quantization=none"%(arch,dataset,gpu,nslice)
        print(cmdLine)
        os.system(cmdLine)
    if index==1:
        arch='alex'
        cmdLine = "python dq.py --arch=%s --dataset=%s --gpu=%d --quantization=none"%(arch,dataset,gpu,nslice)
        print(cmdLine)
        os.system(cmdLine)

def batchNone(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(runNone,(index,gpu))
                      
    pool.close()
    pool.join()
    
    
if __name__ == "__main__":
    nprocess = 14
    dataset='SVHN'
    arch='vgg'
    batchRun(nprocess)

    
