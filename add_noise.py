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
parser.add_argument('--multies', type=int, required=False,default=2, help='the number of mutiple participants')
parser.add_argument('--unit', type=float, required=False, default=0.25,help='the feature ratio held by the attacker')

args = parser.parse_args()
# setting for multi-participant VFl
unit = args.unit
multies = args.multies
other_unit = (1-args.unit)/(args.multies-1)

def add_noise(vec,normal_vecs): # noise scheme for two-split VFL
    avg_value = torch.mean(normal_vecs,dim=0).reshape((-1))
    con = torch.where(avg_value < 0.001)[0]

    size = vec.size()
    vec = vec.reshape((-1))

    vec = vec.clamp_(0, 2.5)
    vec *= 1.15

    gauss_noise_big = torch.normal(mean=0, std=0.5, size=vec.size()).cuda()
    gauss_noise_small = torch.normal(mean=0, std=0.1, size=vec.size()).cuda()

    condition = torch.randn(vec.size()).cuda()
    zeros = torch.zeros_like(vec).cuda()
    replace = torch.where(condition < 0.8, zeros, vec + gauss_noise_small)
    vec = torch.where(vec < 0.4, replace, vec + gauss_noise_big)
    vec = vec.clamp_(0).reshape((size[0],-1))
    vec[:, con] = 0

    return vec.reshape(size)

def add_noise_multi(vec,normal_vecs): # noise scheme for 4-participant VFL
    avg_value = torch.mean(normal_vecs,dim=0).reshape((-1))
    con = torch.where(avg_value < 0.001)[0]

    size = vec.size()
    vec = vec.reshape((-1))

    vec = vec.clamp_(0, 2.5)
    vec *= 1.15

    gauss_noise_big = torch.normal(mean=0, std=0.2, size=vec.size()).cuda()
    gauss_noise_small = torch.normal(mean=0, std=0.05, size=vec.size()).cuda()

    condition = torch.randn(vec.size()).cuda()
    zeros = torch.zeros_like(vec).cuda()
    replace = torch.where(condition < 0.8, zeros, vec + gauss_noise_small)
    vec = torch.where(vec < 0.4, replace, vec + gauss_noise_big)
    vec = vec.clamp_(0).reshape((size[0],-1))
    vec[:, con] = 0
    
    return vec.reshape(size)

def attack_model(model, dataloader, vec_arr,label):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        vec1 = torch.Tensor(np.repeat([vec_arr],B,axis=0)).cuda()
        x_list = x_in.split([int(x_in.size()[2]*unit)]+[int(x_in.size()[2]*other_unit) for i in range(multies-2)]+[x_in.size()[2]-int(x_in.size()[2]*unit)-(multies-2)*int(x_in.size()[2]*other_unit)],dim=2)    
        vec_normal = model.models[0](x_list[0])
        if multies == 2:
            vec1 = add_noise(vec1,vec_normal[:20])
        elif multies > 2:
            vec1 = add_noise_multi(vec1,vec_normal[:20])
        vec = torch.cat([vec1]+[model.models[i](x_list[i]) for i in range(1,multies)], dim=2)

        pred = model.top(vec)
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
        tot = tot + B

    return cum_acc / tot

if __name__ == '__main__':

    GPU = True
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE = 500
    N_EPOCH = 100
    transform_for_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=True,
                                           transform=transform_for_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    is_binary = False
    need_pad = False
    from cnn_model_multi import Model

    input_size = (3, 32, 32)
    class_num = 10
    model = Model(gpu=GPU,multies=multies,unit=unit)

    for label in [0]:#range(class_num):
        atk_list = []
        for dup in [0]:#range(10):
            model.load_state_dict(torch.load('poison_label_%d-%s-%s-%d.model' % (dup,multies,unit,label)))
            target_vec = np.load('label_%d-%s-%s-%d_vec.npy'%(dup,multies,unit,label))
            atkacc = attack_model(model, testloader, target_vec, label)
            atk_list.append(atkacc)
        print('target label: %d, average atk acc: %.4f'%(label,sum(atk_list)/len(atk_list)))