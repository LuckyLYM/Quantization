#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# 15 index
gpu_list=[0,1,2,3,4,5,6,7,1,2,3,4,5,6,7]
archs=['lenet','alex','vgg','resnet','nin']
datasets=['SVHN','CIFAR10','CIFAR100']


def run(index,gpu):  

    time.sleep(2*index)

    dataset=datasets[int(index/5)]
    arch=archs[index%5]

    cmdLine = "python dq.py --arch=%s --quantization=%s --dataset=%s --gpu=%d"%(arch,quan,dataset,gpu)
    print(cmdLine)
    os.system(cmdLine)


def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(run,(index,gpu))
                      
    pool.close()
    pool.join()

# python dq.py --arch=nin --dataset=CIFAR100 --gpu=2 --quantization=none

    
if __name__ == "__main__":
    nprocess = 15

    quan='bwn'
    batchRun(nprocess)

    quan='data'
    batchRun(nprocess)
    
