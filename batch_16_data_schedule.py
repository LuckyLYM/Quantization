#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# nslice form 1 to 7
gpu_list=[5,6,7]
gpu_list2=[5,6,7,5,6,7]
schedule_list=['batch_mix','batch_increase','batch_decrease','epoch_mix','epoch_increase','epoch_decrease']
dataset=''
arch=''


def run(index,gpu,layer_list):  

    layer=layer_list[0]
    schedule=schedule_list[index]
    epoch=200
    #print(dataset,gpu,epoch,layer,schedule)
    # python dq.py --quantization=data --arch=DARTS --dataset=CIFAR10 --gpu=5 --layers=6 --epochs=1 --data_schedule=epoch_mix
    cmdLine = "python dq.py --quantization=data --arch=%s --dataset=%s --gpu=%d --epochs=%d --layers=%d --data_schedule=%s"%(arch,dataset,gpu,epoch,layer,schedule)

    print(cmdLine)
    os.system(cmdLine)


def batchRun(nprocess,layer_list):
    pool = multiprocessing.Pool(processes = nprocess)
    for index,gpu in enumerate(gpu_list2):  
        pool.apply_async(run,(index,gpu,layer_list))

    pool.close()
    pool.join()

    
if __name__ == "__main__":
    

    arch='DARTS'
    dataset_list=['CIFAR10']
    for d in dataset_list:
        dataset=d
        batchRun(6,[6])
    
