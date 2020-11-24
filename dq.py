from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import random

import nin
import alex
import lenet
import vgg
import resnet
import genotypes
#from DARTS import NetworkCIFAR as Network
import DARTS
import others


from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler

def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, os.path.join(model_path,model_name))

            ############################# DATA QUANTIZATION #################################

def quantizeLevel(bacth_progress,epoch_progress,last_slice):

    precision_range=list(range(args.min_bit,args.max_bit+1))
    nlevel=len(precision_range)
    if args.data_schedule=='none':
        return nslice
    elif args.data_schedule=='batch_mix':
        return random.choice(precision_range)
    elif args.data_schedule=='batch_increase':
        index=int(nlevel*bacth_progress)
        return precision_range[index]
    elif args.data_schedule=='batch_decrease':
        index=int(nlevel*bacth_progress)
        return precision_range[nlevel-index-1]
    elif args.data_schedule=='epoch_mix':
        #print(bacth_progress)
        if bacth_progress==0:
            #print(bacth_progress, 'is zero')
            return random.choice(precision_range)
        else:
            return last_slice
    elif args.data_schedule=='epoch_increase':
        index=int(nlevel*epoch_progress)
        return precision_range[index]
    elif args.data_schedule=='epoch_decrease':   
        index=int(nlevel*epoch_progress)
        return precision_range[nlevel-index-1]
    elif args.data_schedule=='max_cardinality':
        compression_ratio=args.compression_ratio
        boundary=args.min_bit/args.original_bit 
        if compression_ratio<boundary:
            return args.min_bit
        else:
            basic_unit=1/args.original_bit
            nslice=int(compression_ratio/basic_unit)
            portion=compression_ratio/basic_unit-nslice
            if bacth_progress<=portion:
                return nslice
            else:
                return nslice+1
    else:
        print('unsupported data scheduling strategy: ',args.data_schedule)
        exit(1)

def quantizeData(data, target, bacth_progress, epoch_progress, last_slice):

    global batch_l1
    global batch_l2
    global instance_l1
    global instance_l2   
    global nBatch

    if quanData==False:
        # no data quantization
        data, target = Variable(data.cuda()), Variable(target.cuda())
        return data, target,-1
    else:      # do data quantization
        data, target = data.cuda(), target.cuda()
        nd= data.size()[0]          # 128
        n = data.nelement()         # 128*3*32*32
        n_per=data[0].nelement()    # 3*32*32
        s = data.size()             # [128,3,32,32]


        # new feature: decide the quantization level
        nslice=quantizeLevel(bacth_progress,epoch_progress,last_slice)

        ############################# Strategy 1 #################################
        # strategy 1: 4-dimensional quantization
        # batch l1 error: 331.303528,  batch l2 error: 8.208151, instance l1 error: 2.590424, instance l2 error: -0.008157 (why negative here???)
        d_l=data
        d_a=0
        for i in range(nslice):
            B=d_l.sign()
            a=B.mul(d_l).sum().div(n)
            d_i=a*B
            
            d_a=d_a+d_i
            d_l=d_l-d_i

        ############################# Strategy 2 #################################
        '''
        # strategy 2: 3-dimensional quantization
        # batch l1 error: 1457.736084,  batch l2 error: 86.315941, instance l1 error: 11.388562,  instance l2 error: 0.078537
        data=data.view(nd,-1)  # [128,3072]
        d_l=data
        d_a=0
        for i in range(nslice):
            B=d_l.sign()      # [128,3072]
            a=B.mul(d_l).sum(dim=1,keepdim=True).div(n_per)  # [128,1]
            d_i=B*a # [128,3072]

            d_a=d_a+d_i
            d_l=d_l-d_i

        # reshape the data
        d_a=d_a.view(s)
        '''
        ############################# Strategy 2 #################################

        ############################# Reconstruction Error #######################
        if train==True:
            nBatch=nBatch+1
            # estimate batch reconstruction error
            d_e=d_l.view(1,-1)
            b_l1= torch.norm(d_e, p=1, dim=1)
            b_l2= torch.norm(d_e, p=2, dim=1)**2
            
            # estimate per data instance reconstruction error
            d_e=d_l.view(nd,-1)
            i_l1= torch.norm(d_e, p=1, dim=1).sum().div(nd)
            tmp=torch.norm(d_e, p=2, dim=1)
            i_l2= (tmp.mul(tmp)).sum().div(nd)     

            # calculate the average l1 and l2 error
            batch_l1=batch_l1+(b_l1-batch_l1)/nBatch
            batch_l2=batch_l2+(b_l2-batch_l2)/nBatch
            instance_l1=instance_l1+(i_l1-instance_l1)/nBatch
            instance_l2=instance_l2+(i_l2-instance_l2)/nBatch

        d_a, target= Variable(d_a), Variable(target)

        return d_a,target,nslice

def train(epoch,trainloader, model, criterion, optimizer):
    model.train()


    last_slice=-1
    
    for batch_idx, (data, target) in enumerate(trainloader):

        if quanModel==True:
            bin_op.binarization()

        bacth_progress=batch_idx/len(trainloader)
        epoch_progress=epoch/args.epochs

        data,target,current_slice=quantizeData(data,target,bacth_progress,epoch_progress,last_slice)
        last_slice=current_slice
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
    
        if quanModel==True:
            bin_op.restore()
            bin_op.updateBinaryGradWeight()
        
        optimizer.step()

        if batch_idx % 10000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test(epoch, testloader, model, criterion):
    global best_acc
    global q_best_acc
    global start

    model.eval()
    test_loss = 0
    correct = 0
    q_test_loss = 0
    q_correct = 0


    if quanModel==True:
        bin_op.binarization()

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    if quanModel==True:
        bin_op.restore()

    '''
    # we don't test the quantized data performance now
    nd= data.size()[0]          # 128
    n = data.nelement()         # 128*3*32*32
    n_per=data[0].nelement()    # 3*32*32
    s = data.size()             # [128,3,32,32]

    d_l=data.clone()
    d_a=0
    for i in range(nslice):
        B=d_l.sign()
        a=B.mul(d_l).sum().div(n)
        d_i=a*B
        
        d_a=d_a+d_i
        d_l=d_l-d_i

    d_a=Variable(d_a.cuda())

    # quantized test data
    q_output = model(d_a)
    q_test_loss += criterion(q_output, target).data.item()
    q_pred = q_output.data.max(1, keepdim=True)[1]
    q_correct += q_pred.eq(target.data.view_as(q_pred)).cpu().sum()
    '''

    ################################# MODEL ACCURACY ##############################
    acc = 100. * float(correct) / len(testloader.dataset)
    q_acc = 100. * float(q_correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        if args.save_model==True:
            save_state(model, best_acc)
    if q_acc> q_best_acc:
        q_best_acc = q_acc
    
    test_loss /= len(testloader.dataset)
    q_test_loss /= len(testloader.dataset)
    duration=time.time()-start

    #print('Test set: Time: {:.2f} instance_l1: {:.8f} instance_l2: {:.8f} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) Average quantized loss: {:.4f}, Quantized Accuracy: {}/{} ({:.2f}%)\n'.format(duration, instance_l1, instance_l2, test_loss * args.batch_size, correct, len(testloader.dataset), 100. * float(correct) / len(testloader.dataset), q_test_loss * args.batch_size, q_correct, len(testloader.dataset), 100. * float(q_correct) / len(testloader.dataset)))

    print('Epoch: {} Duration: {:.2f} Current Accuracy: {:.2f} Best Accuracy: {:.2f}% Best Quantized Accuracy {:.2f}%\n'.format(epoch,duration,acc,best_acc,q_best_acc))

    f.write('Test set: Time: {:.2f} instance_l1: {:.8f} instance_l2: {:.8f} Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%) Average quantized loss: {:.4f}, Quantized Accuracy: {}/{} ({:.4f}%)\n'.format(duration, instance_l1, instance_l2, test_loss * args.batch_size, correct, len(testloader.dataset), 100. * float(correct) / len(testloader.dataset), q_test_loss * args.batch_size, q_correct, len(testloader.dataset), 100. * float(q_correct) / len(testloader.dataset)))

    return

def adjust_learning_rate(optimizer, epoch):
    #update_list = [120, 200, 240, 280]
    update_list = [40, 80, 140, 180]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


# python dq.py --quantization=none --gpu=7
# python dq.py --quantization=joint_bwn --epochs=1 --gpu=7 --dataset=CIFAR100 --arch=alex
# python dq.py --quantization=joint_bwn --epochs=1 --gpu=6 --dataset=SVHN --arch=alex
# python dq.py --quantization=joint_bwn --epochs=1 --gpu=6 --dataset=CIFAR10 --arch=lenet
# python dq.py --quantization=none --gpu=6 --dataset=CIFAR10 --arch=nin

# python dq.py --quantization=none --gpu=5 --dataset=CIFAR10 --arch=alex
# python dq.py --quantization=bwn --gpu=4 --dataset=CIFAR10 --arch=alex
# python dq.py --quantization=data --gpu=3 --dataset=CIFAR10 --arch=alex
# python dq.py --quantization=joint_bwn --gpu=2 --dataset=CIFAR10 --arch=alex
# python dq.py --quantization=data --gpu=3 --dataset=CIFAR10 --arch=DARTS --layers=1 --epochs=2


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', default='data',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',        
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.001',
            help='the intial learning rate')
    parser.add_argument('--gpu', type=int, default=0, 
            help='gpu device id')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--quantization', type=str, default='joint_bwn',
            help='quantization type')    
    parser.add_argument('--slice', type=int, default=16, 
            help='the number of quantized data slice')
    parser.add_argument('--epochs', type=int, default=200, 
            help='the number of traing epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
            help='batch size')
    parser.add_argument('--dataset', action='store', default='none',
            help='dataset')
    parser.add_argument('--binarize_first_layer', action='store_true', default=False, help='')
    parser.add_argument('--binarize_last_layer', action='store_true', default=False, help='')
    parser.add_argument('--save_model', action='store_true', default=False, help='')
    parser.add_argument('--normal_arch', action='store_true', default=False, help='')
    parser.add_argument('--layers', type=int, default=-1, help='total number of layers')    
    parser.add_argument('--train_split', type=float, default=1, help='the portion of dataset used for model training')
    parser.add_argument('--min_bit', type=int, default=3, help='the minimum bits for network binarization')
    parser.add_argument('--max_bit', type=int, default=8, help='the maximum bits for network binarization')
    parser.add_argument('--original_bit', type=int, default=8, help='the number of bits for the original dataset')
    parser.add_argument('--compression_ratio', type=float, default=1, help='the portion of dataset we used for model training')
    parser.add_argument('--data_schedule', type=str, default='none',help='the scheduling of training data')
    parser.add_argument('--lr_scheduler', type=str, default='Adam',help='the learning rate scheduler')

    # max_cardinality, max_precision
    # use the normal architecture instead of that specfic designed for binary network
    # quantization:
    # bwn => only model quantization
    # joint_bwn => joint data and model quantization
    # none => no model and no data quantization
    # data => only data quantization

    args = parser.parse_args()
    print('==> Options:',args)

    start=time.time()
    nslice=args.slice
    batch_l1=0
    batch_l2=0
    instance_l1=0
    instance_l2=0
    nBatch=0

    quan=args.quantization
    arch=args.arch
    quanData=True
    quanModel=True

    if quan!='data' and quan!='joint_bwn' and quan!='joint_xnor':
        quanData=False
    if quan=='data' or quan=='none':
        quanModel=False

    print("Training Mode: ",quanData,quanModel)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.set_device(args.gpu)

    ############################### DATASET #################################
    dataset=args.dataset
    train_split=args.train_split
    datapath=os.path.join(dataset,args.data)
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    
    log_path=os.path.join(dataset,'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    model_path=os.path.join(dataset,'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    # the full training set is 5w images
    # how many data we used for training

    if dataset=='CIFAR10':
        train_transform, valid_transform = data._data_transforms_cifar10()
        trainset = dset.CIFAR10(root=datapath, train=True, download=True, transform=train_transform)
        testset = dset.CIFAR10(root=datapath, train=False, download=True, transform=valid_transform)
        input_dim=32*32*3
        num_classes=10
    elif args.dataset=='CIFAR100':
        train_transform, valid_transform = data._data_transforms_cifar10()
        trainset = dset.CIFAR100(root=datapath, train=True, download=True, transform=train_transform)
        testset = dset.CIFAR100(root=datapath, train=False, download=True, transform=valid_transform)
        input_dim=32*32*3
        num_classes=100
    elif args.dataset=='SVHN':
        train_transform, valid_transform = data._data_transforms_SVHN()
        trainset = dset.SVHN(root=datapath, split='train', download=True, transform=train_transform)
        testset = dset.SVHN(root=datapath, split='test', download=True, transform=valid_transform)
        input_dim=32*32*3
        num_classes=10
    elif args.dataset=='MNIST':
        trainset = dset.MNIST(root=datapath, train=True, transform=transforms.ToTensor(),download=True)
        testset = dset.MNIST(root=datapath, train=False, transform=transforms.ToTensor(),download=True)
        input_dim=28*28
        num_classes=10
    else:
        print('invalid dataset ',dataset)
        exit(1)

    dataset_size=len(trainset)
    indices=list(range(dataset_size))


    ############################## Two Data Selection Strategies ##############################
    if args.data_schedule=='max_cardinality':
        compression_ratio=args.compression_ratio
        boundary=args.min_bit/args.original_bit 
        if compression_ratio<boundary:
            train_split=compression_ratio/boundary
            nslice=args.min_bit
            #print(args.data_schedule,args.compression_ratio,train_split,nslice,quanData)

        else:
            train_split=1
            basic_unit=1/args.original_bit
            nslice=int(compression_ratio/basic_unit)
            portion=compression_ratio/basic_unit-nslice

            # portion:      nslice
            # 1-portion:    nslice+1

            #print(args.data_schedule,args.compression_ratio,train_split,nslice,portion,quanData)

    elif args.data_schedule=='max_precision':
        train_split=args.compression_ratio
        # an exception here, we don't quantize the input training data
        quanData=False
        #print(args.data_schedule,args.compression_ratio,train_split,quanData)


    split=int(np.floor(train_split*dataset_size))
    train_indices=indices[:split]
    train_sampler=SubsetRandomSampler(train_indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=False, num_workers=2,pin_memory=True,sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
            shuffle=False, num_workers=2,pin_memory=True)



    ############################### MODEL #################################
    print('==> building model',args.arch,'...')
    supported_model_list=['nin','alex','lenet','vgg','resnet']

    if args.normal_arch==True:
        if arch in supported_model_list:
            model=eval(arch+'.normal(num_classes)')
        else:
            print('unsupported arch ',arch)
            exit(1)
        arch='normal_'+arch 

    elif quan=='bwn' or quan=='joint_bwn' or quan=='none' or quan=='data':     # BWN net
        if arch in supported_model_list:
            model=eval(arch+'.BWNNet(num_classes)')
        elif arch=='DARTS':
            # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
            genotype = eval("genotypes.%s" % args.arch)
            if args.dataset=='MNIST':
                model = DARTS.NetworkMNIST(12, num_classes, args.layers, False, genotype)
            else:
                model = DARTS.NetworkCIFAR(16, num_classes, args.layers, False, genotype)
            model.drop_path_prob=0
        elif arch=='linear':
            model=others.LinearRegression(input_dim,num_classes)
        elif arch=='logistic':
            model=others.LogisticRegression(input_dim,num_classes)
        elif arch=='mlp':
            model=others.MLP(input_dim,num_classes)
        else:
            print('unsupported arch ',arch)
            exit(1)

    elif quan=='xnor' or quan=='joint_xnor':    # XNOR net
        if arch in supported_model_list:
            model=eval(arch+'.XNORNet(num_classes)')
        else:
            print('unsupported arch ',arch)
            exit(1)

    else:
        print('unsupported quantization ',quan)
        exit(1)

    ############################### LOG FILE PATH #################################

    if quan == 'bwn' or quan == 'xnor' or quan=='none':
        model_name=arch+'_'+quan+'.pth.tar'
        if arch=='DARTS':
            log_name='log_'+arch+"_"+str(args.layers)+'_'+quan
        else:
            log_name='log_'+arch+'_'+quan
        f=open(os.path.join(log_path,log_name),'w')

    elif quan =='data':
        if arch=='DARTS':
            if args.data_schedule=='none':
                model_name=arch+"_"+str(args.layers)+"_data_"+str(args.slice)
            elif args.data_schedule=='max_precision' or args.data_schedule=='max_cardinality':
                model_name=arch+"_"+str(args.layers)+"_data_"+args.data_schedule+'_'+str(args.compression_ratio)
            else:
                model_name=arch+"_"+str(args.layers)+"_data_"+args.data_schedule
        else:
            if args.data_schedule=='none':
                model_name=arch+"_data_"+str(args.slice)
            elif args.data_schedule=='max_precision' or args.data_schedule=='max_cardinality':
                model_name=arch+"_data_"+args.data_schedule+'_'+str(args.compression_ratio)
            else:
                model_name=arch+"_data_"+args.data_schedule

        # implemented for train_split
        if int(args.train_split)!=1:
            model_name=model_name+'_'+str(args.train_split)

        log_name='log_'+model_name
        #print(log_name)
        model_name=model_name+'.pth.tar'
        f=open(os.path.join(log_path,log_name),'w')  

    elif quan == 'joint_xnor' or quan=='joint_bwn':
        model_name=arch+"_"+quan+"_"+str(args.slice)
        log_name='log_'+model_name
        if args.binarize_first_layer==True:
            model_name=model_name+"_bf"
        if args.binarize_last_layer==True:
            model_name=model_name+"_bl"
        model_name=model_name+'.pth.tar'
        f=open(os.path.join(log_path,log_name),'w')

    else:
        raise Exception(args.quantization+' is currently not supported')


    ############################### MODEL INTIALIZATION ###############################
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        q_best_acc=0
        if arch!='resnet' and arch!='DARTS':
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.05)
                    m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    model.cuda()
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    #print(model)


    ############################### OPTIMIZER ###############################
    if args.lr_scheduler=='Adam':
        base_lr = float(args.lr)
        param_dict = dict(model.named_parameters())
        params = []
        for key, value in param_dict.items():
            params += [{'params':[value], 'lr': base_lr,
                'weight_decay':0.00001}]

        optimizer = optim.Adam(params, lr=base_lr,weight_decay=0.00001)

    elif args.lr_scheduler=='SGD':
        optimizer = torch.optim.SGD(
          model.parameters(),
          0.001,
          momentum=0.9,
          weight_decay=3e-4)
    
    else:
        print('unsupported learning rate scheduler: ',args.lr_scheduler)
        exit(1)

    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    # quantization:
    # bwn => only model quantization
    # joint_bwn => joint data and model quantization
    # none => no model and no data quantization
    # data => only data quantization
    if quanModel==True:
        bin_op = util.BinOp(model,binarize_first_layer=args.binarize_first_layer,binarize_last_layer=args.binarize_last_layer)

    # do the evaluation if specified
    if args.evaluate:
        test(testloader,model,criterion)
        exit(0)


    ############################### MODEL TRAINING ###############################
    for epoch in range(0, args.epochs):
        if args.lr_scheduler=='Adam':
            adjust_learning_rate(optimizer, epoch)
        train(epoch,trainloader,model,criterion,optimizer)
        test(epoch, testloader,model,criterion)