from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset

from models import nin
from torch.autograd import Variable
from models import alexnet



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
    torch.save(state, os.path.join('models',model_name))

def train(epoch,trainloader, model, criterion, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):

        # process the weights including binarization
        # quantize the conv weights, and store the full-precision weights
        if args.quantization!='none':
            bin_op.binarization()
        
        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore the full precision weights
        if args.quantization!='none':
            bin_op.restore()
            bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test(testloader, model, criterion):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    
    if args.quantization!='none':
        bin_op.binarization()

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    if args.quantization!='none':
        bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

    f.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))

    return

def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--gpu', type=int, default=0, 
        help='gpu device id')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--quantization', type=str, default='bwn', help='quantization type')    # should modify

    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.set_device(args.gpu)


    # root-> train_data/train_labels
    #trainset = data.dataset(root=args.data, train=True)
    #testset = data.dataset(root=args.data, train=False)

    # must have data preprocessing to transfer from PIL images to nparray
    train_transform, valid_transform = data._data_transforms_cifar10()
    trainset = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    testset = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')


    if args.quantization == 'bwn':
        model = nin.BWNNet()
        model_name=args.arch+'_bwn.pth.tar'
        f=open(os.path.join('log','log_'+args.arch+"_bwn"),'w')

    elif args.quantization == 'xnor':
        model = nin.XNORNet()
        model_name=args.arch+'_xnor.pth.tar'
        f=open(os.path.join('log','log_'+args.arch+"_xnor"),'w')

    elif args.quantization == 'joint_bwn':

        if args.arch=='nin':
            model = nin.BWNNet()
        elif args.arch=='alex':
            model=alexnet.AlexNet()

        name=args.arch+"_joint_bwn_"+str(args.slice)
        if args.binarize_first_layer==True:
            name=name+"_bf"
        if args.binarize_last_layer==True:
            name=name+"_bl"
        model_name=name+'.pth.tar'
        f=open(os.path.join('log','log_'+name),'w')

    elif args.quantization == 'joint_xnor':
        model = nin.XNORNet()
        name=args.arch+"_joint_xnor_"+str(args.slice)
        if args.binarize_first_layer==True:
            name=name+"_bf"
        if args.binarize_last_layer==True:
            name=name+"_bl"
        model_name=name+'.pth.tar'
        f=open(os.path.join('log','log_'+name),'w')

    elif args.quantization=='none':
        model = nin.BWNNet()
        model_name=args.arch+'_none.pth.tar'
        f=open(os.path.join('log','log_'+args.arch+"_none"),'w')
    else:
        raise Exception(args.quantization+' is currently not supported')



    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
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

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

    optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    if args.quantization!='none':
        bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        test(testloader,model,criterion)
        exit(0)

    # start training
    for epoch in range(1, 320):
        adjust_learning_rate(optimizer, epoch)
        train(epoch,trainloader,model,criterion,optimizer)
        test(testloader,model,criterion)


    # save model here
    # they already do the model saving in the model test process
    #torch.save(model.state_dict(), "models/best.pt")
    '''
    torch.save({
    'epoch': 320,
    'model_state_dict': model.state_dict(),
    'best_acc' : best_acc,
    },os.path.join("models",args.arch))
    '''