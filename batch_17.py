#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


gpu_list=[0,1,2,3,0,1,2,3]
slice_list=[13,14,27,28,29,30,31,32]

# 14 threads, 7 gpus
# 1-28
def run(index,gpu):  
    arch='nin'
    '''
    begin=index*2+1
    end=(index+1)*2+1
    '''

    time.sleep(10*index)
    '''
    for nslice in range(begin,end):
    '''
    nslice=slice_list[index]
    cmdLine = "python dq.py --arch=%s --gpu=%d --slice=%d"%(arch, gpu, nslice)
    print(cmdLine)
    os.system(cmdLine)

    '''
    for i in range(begin,end):
        cmdLine = "python dq.py --arch=%s --binarize_first_layer --gpu=%d --slice=%d"%(arch, gpu, nslice)
        print(cmdLine)
        os.system(cmdLine)
    '''

def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(run,(index,gpu))
                      
    pool.close()
    pool.join()
    
    
if __name__ == "__main__":
    nprocess = 8
    batchRun(nprocess)
    
