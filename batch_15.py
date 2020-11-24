#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


gpu_list=[1,2,3,4,5,6,7,1,2,3,4,5,6,7]


# 14 threads, 7 gpus
# 1-28
def run(index,gpu):  
    begin=index*2+1
    end=(index+1)*2+1
    
    time.sleep(5*index)
    for nslice in range(begin,end):
        cmdLine = "python dq.py --arch=%s --dataset=%s --gpu=%d --slice=%d"%(arch,dataset,gpu,nslice)
        print(cmdLine)
        os.system(cmdLine)


def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(run,(index,gpu))
                      
    pool.close()
    pool.join()



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
    arch='alex'
    batchRun(nprocess)

    arch='nin'
    batchRun(nprocess)

    nprocess=2
    batchNone(nprocess)

    
