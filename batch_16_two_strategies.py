#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# nslice form 1 to 7
gpu_list=[4,5,6,7]
gpu_list2=[4,5,6,7,4,5,6,7]
rate_list=[0.2,0.4,0.6,0.8,0.8,0.6,0.4,0.2]
schedule_list=['max_precision','max_cardinality']
dataset=''
arch=''


def run(index,gpu,layer_list):  

    layer=layer_list[0]
    if index<4:
        schedule=schedule_list[0]
    else:
        schedule=schedule_list[1]
    epoch=100
    #print(dataset,gpu,epoch,layer,schedule)
    # python dq.py --quantization=data --arch=DARTS --dataset=CIFAR10 --gpu=5 --layers=6 --epochs=1 --data_schedule=epoch_mix
    rate=rate_list[index]
    cmdLine = "python dq.py --quantization=data --arch=%s --dataset=%s --gpu=%d --epochs=%d --layers=%d --data_schedule=%s --compression_ratio=%f"%(arch,dataset,gpu,epoch,layer,schedule,rate)

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
    dataset_list=['CIFAR10','SVHN','CIFAR100']
    #dataset_list=['CIFAR100']
    for d in dataset_list:
        dataset=d
        batchRun(8,[6])
    
