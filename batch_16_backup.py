#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


gpu_list=[0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]


# 14 threads, 7 gpus
# 1-28
def run(index,gpu):  
    arch='alex'
    begin=index*2+1
    end=(index+1)*2+1
    time.sleep(5*index)

    for nslice in range(begin,end):
        cmdLine = "python dq.py --arch=%s --gpu=%d --slice=%d --epochs=10"%(arch, gpu, nslice)
        print(cmdLine)
        os.system(cmdLine)
  



def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(run,(index,gpu))
                      
    pool.close()
    pool.join()
    
    
if __name__ == "__main__":
    nprocess = 16
    batchRun(nprocess)
    
