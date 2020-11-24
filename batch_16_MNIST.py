#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# nslice form 1 to 7
#gpu_list=[4,5,6,7]
gpu_list2=[4,5,6,7,4,5,6,7]
rate_list=[0.2,0.8,0.4,0.6,0.8,0.6,0.4,0.2]
schedule_list=['max_precision','max_cardinality']
dataset=''
arch=''


def run(index,gpu):  

    epoch=1
    if index<4:
        schedule=schedule_list[0]
    else:
        schedule=schedule_list[1]
    
    rate=rate_list[index]
    cmdLine = "python dq.py --quantization=data --arch=DARTS --layers=3 --lr=0.001 --dataset=MNIST --gpu=%d --epochs=50 --data_schedule=%s --compression_ratio=%f"%(gpu,schedule,rate)
      
    print(cmdLine)
    os.system(cmdLine)

    #print(dataset,gpu,epoch,layer,schedule)
    # python dq.py --quantization=data --arch=linear --dataset=MNIST --gpu=5 --epochs=1 --data_schedule=max_cardinality --compression_ratio=0.2




def batchRun(nprocess):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list2):  
        pool.apply_async(run,(index,gpu))

    pool.close()
    pool.join()

    
if __name__ == "__main__":
    


    batchRun(8)

    cmdLine = "python dq.py --quantization=none --arch=DARTS --layers=3 --lr=0.001 --dataset=MNIST --gpu=5 --epochs=50"       
    
    print(cmdLine)
    os.system(cmdLine)
    
