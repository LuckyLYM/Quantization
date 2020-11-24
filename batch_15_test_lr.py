#!/usr/bin/env python
import os
import sys
import multiprocessing
import pickle
import time


# nslice form 1 to 7
#gpu_list=[4,5,6,7]
gpu_list2=[1,2,3,4,5,6,7,1]
rate_list=[0.2,0.8,0.4,0.6,0.8,0.6,0.4,0.2]
schedule_list=['max_cardinality','max_precision']
dataset=''
arch=''


def run(index,gpu):  

    epoch=50

    if index<8:
        if index<4:
            schedule=schedule_list[0]
        else:
            schedule=schedule_list[1]
        
        rate=rate_list[index]
        cmdLine = "python dq.py --quantization=data --arch=%s --dataset=%s --gpu=%d --epochs=%d --data_schedule=%s --compression_ratio=%f"%(arch,dataset,gpu,epoch,schedule,rate)
    else:
        cmdLine = "python dq.py --quantization=none --arch=%s --dataset=%s --gpu=%d --epochs=%d"%(arch,dataset,gpu,epoch)       

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
    

    dataset_list=['CIFAR10']
    arch_list=['linear']
    #arch_list=['mlp']
    #dataset_list=['CIFAR100']
    for d in dataset_list:
        dataset=d
        for a in arch_list:
            arch=a

            batchRun(8)
    
