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
import cal_centers as cc
import generate_vec as gv
import search_vec as sv
import warnings

import matplotlib.pyplot as plt

import matplotlib.lines as lines
from matplotlib.ticker import  FuncFormatter

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=int, required=True, help='the target class of your attack')
parser.add_argument('--dup', type=int, required=True, help='the ID for duplicated models of a same setting')
parser.add_argument('--magnification',type=int,required=True,help='the size of the auxiliary set will be 50*magnification')
parser.add_argument('--multies', type=int, required=False,default=2, help='the number of mutiple participants')
parser.add_argument('--unit', type=float, required=False, default=0.25,help='the feature ratio held by the attacker')
parser.add_argument('--clean-epoch', type=int, required=False,default=80, help='the number of training epochs without poisoning')

args = parser.parse_args()
other_unit = (1-args.unit)/(args.multies-1)

target_num = 50
normal_num = 50
clean_epoch = args.clean_epoch

def prepared_data(set):
    data = []
    label = []
    for idx in range(len(set)):
        x,y = set[idx]
        data.append({'id':idx,'data':x})
        label.append(y)

    return data,label

class CIFAR10(torch.utils.data.Dataset):
    def __init__(self,data,label,transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, item):
        x = self.data[item]['data']
        if not(self.transform is None):
            x = self.transform(x)
        y = self.label[item]
        id = self.data[item]['id']

        return x,y,id

    def __len__(self):
        return len(self.label)


def steal_samples(trn_x,trn_y,t):
    targets = []
    for idx in range(len(trn_y)):
        if trn_y[idx] == t:
            targets.append(trn_x[idx]['id'])
    num = target_num*target_magnification
    print("clean image used for class %d: %d"%(t,num))

    steal_id = random.sample(targets,num)
    data = []
    label = []
    for idx in steal_id:
        data.append(trn_x[idx])
        label.append(trn_y[idx])

    steal_set = CIFAR10(data,label,transform=transform_for_train)
    steal_id = torch.tensor(steal_id)

    return steal_set,steal_id

def design_vec(class_num,model,label,steal_set):
    target_clean_vecs = gv.generate_target_clean_vecs(model.models[0],steal_set,args.unit,bottom_series=0)
    
    dim = filter_dim(target_clean_vecs)

    center = cc.cal_target_center(target_clean_vecs[dim].copy(),kernel_bandwidth=1000) 
    
    target_vec = sv.search_vec(center,target_clean_vecs,args.unit)
    
    target_vec = target_vec.reshape((64,int((int((int(32*args.unit)-2)/2+1)-2)/2+1),8))
    
    return target_vec

def filter_dim(vecs):
    coef = np.corrcoef(vecs)
    rows = np.sum(coef,axis=1)
    selected = np.argpartition(rows,-target_num)[-target_num:]
    print(np.mean(np.corrcoef(vecs[selected])))
    return selected

def train_model(model, dataloader,label,steal_set,steal_id,epoch_num,start_epoch=0,is_binary=False, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

    for epoch in range(start_epoch,epoch_num):
        t1 = time.time()

        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0

        if epoch >= clean_epoch and epoch%10 == 0:
            vec_arr = design_vec(class_num, model, label, steal_set)
        
        for i, (x_in, y_in, id_in) in enumerate(dataloader):
            B = x_in.size()[0]

            if args.unit != 1:
                x_list = x_in.split([int(x_in.size()[2]*args.unit)]+[int(x_in.size()[2]*other_unit) for i in range(args.multies-2)]+[x_in.size()[2]-int(x_in.size()[2]*args.unit)-(args.multies-2)*int(x_in.size()[2]*other_unit)],dim=2)
            else:
                x_list = [x_in]

            vec1 = model.models[0](x_list[0])

            if epoch >= clean_epoch:
                condition = []
                for idx in range(B):
                    if id_in[idx] in steal_id:
                        condition.append(idx)
                vec1[condition] = torch.tensor(vec_arr).cuda()

            if args.unit != 1:
                vec = torch.cat([vec1]+[model.models[i](x_list[i]) for i in range(1,args.multies)], dim=2)
            else:
                vec = vec1

            pred = model.top(vec)
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
            t2 = time.time()
            print("Epoch %d, loss = %.4f, acc = %.4f (%.4fs)" % (epoch, cum_loss / tot, cum_acc / tot,t2-t1))
        
    return vec_arr


def eval_model(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in,_) in enumerate(dataloader):
        B = x_in.size()[0]
        pred = model(x_in)
        if is_binary:
            cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot

def attack_model(model, dataloader, vec_arr,label,multies,unit,other_unit):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in,_) in enumerate(dataloader):
        B = x_in.size()[0]

        if args.unit != 1:
            x_list = x_in.split([int(x_in.size()[2]*unit)]+[int(x_in.size()[2]*other_unit) for i in range(multies-2)]+[x_in.size()[2]-int(x_in.size()[2]*unit)-(multies-2)*int(x_in.size()[2]*other_unit)],dim=2)
        
        vec1 = torch.Tensor(np.repeat([vec_arr],B,axis=0)).cuda()

        if args.unit != 1:
            vec = torch.cat([vec1]+[model.models[i](x_list[i]) for i in range(1,multies)], dim=2)
        else:
            vec = vec1

        pred = model.top(vec)
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
        tot = tot + B
    return cum_acc / tot


if __name__ == '__main__':
    target_magnification = args.magnification

    GPU = True
    if GPU:
        torch.cuda.manual_seed_all(args.dup)
        random.seed(args.dup)
        torch.manual_seed(args.dup)
        np.random.seed(args.dup)

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
    trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=True)

    trn_x,trn_y = prepared_data(trainset)
    dl_train_set = CIFAR10(trn_x,trn_y,transform=transform_for_train)
    val_x,val_y = prepared_data(testset)
    dl_val_set = CIFAR10(val_x,val_y,transform=transform_for_test)
    is_binary = False
    need_pad = False

    from cnn_model_multi import Model
    
    input_size = (3, 32, 32)
    class_num = 10

    model = Model(gpu=GPU,multies=args.multies,unit=args.unit)
    trainloader = torch.utils.data.DataLoader(dl_train_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(dl_val_set, batch_size=BATCH_SIZE, shuffle=True)

    steal_set,steal_id = steal_samples(trn_x,trn_y,args.label)

    label = args.label
    dup = args.dup

    t1=time.time()

    model.load_state_dict(torch.load('clean-%d-%d-%s.model'%(args.dup,args.multies,args.unit)))

    last_vec_arr = train_model(model, trainloader,label,steal_set,steal_id,epoch_num=N_EPOCH,start_epoch=clean_epoch, is_binary=is_binary,verbose=True)
    torch.save(model.state_dict(),'poison_label_%d-%d-%s-%d.model'%(args.dup,args.multies,args.unit,args.label))
    
    cleanacc = eval_model(model, testloader, is_binary=is_binary)
    print('clean acc: %.4f'%cleanacc)

    atkacc = attack_model(model, testloader, last_vec_arr, label,args.multies,args.unit,other_unit)
    print('target label: %d, attack acc: %.4f' % (label, atkacc))

    np.save('label_%d-%d-%s-%d_vec'%(args.dup,args.multies,args.unit,args.label),last_vec_arr)

    t2 = time.time()
    print("Training a model costs %.4fs."%(t2-t1))
