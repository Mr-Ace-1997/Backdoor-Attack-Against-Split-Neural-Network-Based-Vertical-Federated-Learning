import torch
import torch.nn.functional as F
import numpy as np
import random

def cal_distance(point,direct): 
    points = point.unsqueeze(0).repeat(10,1)
    c = F.normalize(direct,dim=1,p=2)
    norm = torch.div(torch.sum(torch.mul(points,direct),dim=1),torch.norm(direct,dim=1,p=2)).unsqueeze(1)
    e = points-torch.mul(c,norm)

    return torch.norm(e,dim=1,p=2)

def euclidean_dist(pointA, pointB):
    
    total = (pointA - pointB)
    return torch.norm(total,p=2)  

def max_difference(vec,centers,label=None):
    distances = cal_distance(vec,centers)
    if label == None:
        distances,_ = torch.sort(distances)
        first_min = distances[0]
        second_min = torch.mean(distances)
    else:
        first_min = distances[label]
        second_min = torch.mean(distances)

    return (second_min-first_min)/first_min

def forward(point,centers,label):
    max_diff = max_difference(point,centers,label)
    return max_diff
    
def search_vec(center,target_clean_vecs,unit):
    center = torch.tensor(center)

    target_clean_vecs = torch.tensor(target_clean_vecs)
    max_length = torch.max(torch.norm(target_clean_vecs,dim=1,p=2))
    target_vec = center * max_length/torch.norm(center,p=2)*0.85

    target_vec = target_vec.detach().cpu().numpy()

    return target_vec.reshape((64,int((int((int(32*unit)-2)/2+1)-2)/2+1),8))