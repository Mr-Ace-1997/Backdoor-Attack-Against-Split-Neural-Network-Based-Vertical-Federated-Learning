import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from datetime import datetime
import json
import time
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def generate_vec(model,dataloader,unit,bottom_series):
    vecs = []
    for i, (x_in, y_in,_) in enumerate(dataloader):
        if unit != 1:
            x_part = x_in.split([int(x_in.size()[2] * unit), x_in.size()[2] - int(x_in.size()[2] * unit)], dim=2)
            pred = model(x_part[bottom_series]).detach().cpu().numpy()
        else:
            pred = model(x_in).detach().cpu().numpy()

        vecs.append(pred)
    vecs = np.array(vecs)
    shape = vecs.shape
    vecs = vecs.reshape((shape[0] * shape[1], shape[2]*shape[3]*shape[4]))
    return vecs

def generate_all_clean_vecs(class_num,model,testset,unit,bottom_series=0):
    all_clean_vecs = []
    for label in range(class_num):
        target_set = testset[label]
        targetloader = torch.utils.data.DataLoader(target_set, batch_size=1000, shuffle=True)
        vecs = generate_vec(model, targetloader,unit,bottom_series)
        all_clean_vecs.append(vecs)
    return np.array(all_clean_vecs)

def generate_target_clean_vecs(model,testset,unit,bottom_series=0):
    target_set = testset
    targetloader = torch.utils.data.DataLoader(target_set, batch_size=1000, shuffle=True)
    target_clean_vecs = generate_vec(model, targetloader, unit, bottom_series)
    return target_clean_vecs
