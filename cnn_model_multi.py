import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottomModel(nn.Module):
    def __init__(self,gpu=False):
        super(BottomModel,self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if gpu:
            self.cuda()

    def forward(self,x):
        if self.gpu:
            x = x.cuda() 

        x = F.relu(self.conv1(x)) 
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) 
        x = self.max_pool(F.relu(self.conv4(x))) 

        return x

class TopModel(nn.Module):
    def __init__(self,gpu=False,input_size=8):
        super(TopModel,self).__init__()
        self.gpu = gpu
        self.linear = nn.Linear(64*input_size*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)

        if gpu:
            self.cuda()

    def forward(self,x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = F.relu(self.linear(x.view(B,-1)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x

class Model(nn.Module):
    def __init__(self, gpu=False,multies=2,unit = 0.25):
        super(Model, self).__init__()
        self.gpu = gpu
        self.multies = multies
        self.unit = unit
        self.other_unit = (1-unit)/(multies-1)
        self.models = nn.ModuleList([BottomModel(gpu) for i in range(self.multies)])
        self.top = TopModel(gpu,int((int((int(32*self.unit)-2)/2+1)-2)/2+1)+(multies-2)*int((int((int(32*self.other_unit)-2)/2+1)-2)/2+1)+int((int((32-int(32*self.unit)-(multies-2)*int(32*self.other_unit)-2)/2+1)-2)/2+1))

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x_list = x.split([int(x.size()[2]*self.unit)]+[int(x.size()[2]*self.other_unit) for i in range(self.multies-2)]+[x.size()[2]-int(x.size()[2]*self.unit)-(self.multies-2)*int(x.size()[2]*self.other_unit)],dim=2)
        x_list = [self.models[i](x_list[i]) for i in range(self.multies)]
        x = torch.cat(x_list,dim=2)
        x = self.top(x)
        return x
        
    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
