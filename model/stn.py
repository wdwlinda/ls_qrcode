#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:18:36 2019

@author: xingyu
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import sys



class STNet(nn.Module):
    
    def __init__(self):
        super(STNet, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=5),
                nn.MaxPool2d(3, stride=3),
                nn.ReLU(True)
                )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
                nn.Linear(32 * 14 * 2, 32),
                nn.ReLU(True),
                nn.Linear(32, 3*2)
                )
        # Initialize the weights/bias with identity transformation 
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
        
    def forward(self, x):        
        xs = self.localization(x)
        xs = xs.view(-1, 32*14*2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
         
        return x


class STNet_2(nn.Module):
    
    def __init__(self):
        super(STNet_2, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
                )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
                nn.Linear(32 * 20 * 5, 32),
                nn.ReLU(True),
                nn.Linear(32, 3*2)
                )
        # Initialize the weights/bias with identity transformation 
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
        
    def forward(self, x):        
        xs = self.localization(x)
        xs = xs.view(-1, 32*20*5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
         
        return x



sys.path.append(".") 
from utils import summary
 
if __name__ == "__main__":    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = STNet().to(device)    
    # summary(model, input_size=(3, 24, 94), batch_size=1, device="cpu")

    # input = torch.Tensor(2, 3, 24, 94).to(device)
    # output = model(input)
    # print('output shape is', output.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = STNet_2().to(device)    
    summary(model, input_size=(3, 20, 80), batch_size=1, device="cpu")