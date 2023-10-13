import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from datetime import datetime
import json
import argparse
import time
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--clean-epoch', type=int, required=False,default=80, help='the number of training epochs without poisoning')
parser.add_argument('--dup', type=int, required=True, help='the ID for duplicated models of a same setting')
parser.add_argument('--multies', type=int, required=False,default=2, help='the number of mutiple participants')
parser.add_argument('--unit', type=float, required=False, default=0.25,help='the feature ratio held by the attacker')

def train_model(model, dataloader,epoch_num, is_binary, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i, (x_in, y_in) in enumerate(dataloader):
            B = x_in.size()[0]
            pred = model(x_in)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
        if verbose:
            print("Epoch %d, loss = %.4f, acc = %.4f" % (epoch, cum_loss / tot, cum_acc / tot))
    return


def eval_model(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        pred = model(x_in)
        if is_binary:
            cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot

if __name__ == '__main__':
    args = parser.parse_args()

    GPU = True
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE = 500
    N_EPOCH = 100
    
    transform_for_train = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=5),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    transform_for_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True,
                                            transform=transform_for_train)
    testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=True,
                                           transform=transform_for_test)
    is_binary = False
    need_pad = False

    from cnn_model_multi import Model

    input_size = (3, 32, 32)
    class_num = 10

    model = Model(gpu=GPU,multies=args.multies,unit=args.unit)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    if args.clean_epoch:
        t1 = time.time()

        train_model(model, trainloader, epoch_num=args.clean_epoch, is_binary=is_binary, verbose=True)
        torch.save(model.state_dict(),'clean_epoch_%d-%d-%s.model'%(args.dup,args.multies,args.unit))
        
        train_model(model, trainloader, epoch_num=N_EPOCH-args.clean_epoch, is_binary=is_binary, verbose=True)
        cleanacc = eval_model(model, testloader, is_binary=is_binary)
        torch.save(model.state_dict(),'clean-%d-%d-%s.model'%(args.dup,args.multies,args.unit))
        print('clean acc: %.4f' % cleanacc)

        t2 = time.time()
        print("Training a model costs %.4fs." % (t2 - t1))

    else:
        t1 = time.time()

        train_model(model, trainloader, epoch_num=N_EPOCH, is_binary=is_binary,verbose=True)
        cleanacc = eval_model(model, testloader, is_binary=is_binary)
        torch.save(model.state_dict(),'clean%d-%d-%s.model'%(args.dup,args.multies,args.unit))
        print('clean acc: %.4f'%cleanacc)

        t2 = time.time()
        print("Training a model costs %.4fs." % (t2 - t1))



