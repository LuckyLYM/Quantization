#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


gpu_list=[0,1,2,3,4,5,6,7]*4


def run(index,gpu):
    
    time.sleep(index*5)  
    nslice=index+1
    cmdLine = "python dq.py --arch=%s --dataset=CIFAR10 --gpu=%d --slice=%d"%(arch, gpu, nslice)
    print(cmdLine)
    os.system(cmdLine)

def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(run,(index,gpu))
                      
    pool.close()
    pool.join()
    
    
if __name__ == "__main__":
    arch='lenet'
    nprocess = 32
    batchRun(nprocess)

    cmdLine = "python dq.py --arch=%s --dataset=CIFAR10 --quantization=none --gpu=%d --slice=%d"%(arch, 1, nslice)
    print(cmdLine)
    os.system(cmdLine)
    
