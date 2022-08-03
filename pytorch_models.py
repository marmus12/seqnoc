#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:30:46 2022

@author: root
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


    

class SNLM(nn.Module):   # SEQNOC-LM
    def __init__(self,device=torch.device('cuda')):
        super().__init__()
        self.ctx_type=144 #6x6x4
        self.ksize = 6
        self.nhidden1 = 20
        self.nhidden2 = 20
        self.nhidden3 = 40
        self.conv1 = nn.Conv2d(4,self.nhidden1,kernel_size=2,stride=2).to(device) # (W/2)x(W/2)  xself.nhidden1  #3x3x30=
        
        self.conv2 = nn.Conv2d(self.nhidden1,self.nhidden2,kernel_size=2,stride=1).to(device) # (W/2)-1 xself.nhidden2 #2x2x10=240
        
        self.conv3 = nn.Conv2d(self.nhidden2,self.nhidden3,kernel_size=2,stride=1).to(device) #(W/2)-2 xself.nhidden2 #1x1x80


        self.conv_out = nn.Conv2d(self.nhidden3, 16, 1).to(device) #16x(W/2)-1 


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))      
        x = F.softmax(self.conv_out(x),dim=1).squeeze()
        return x

class SNLM2(nn.Module):  
    def __init__(self,device=torch.device('cuda')):
        super().__init__()
        self.ctx_type=144 #6x6x4
        self.ksize = 6
        self.nhidden1 = 10
        self.nhidden2 = 10
        self.nhidden3 = 20
        self.conv1 = nn.Conv2d(4,self.nhidden1,kernel_size=2,stride=2).to(device) # (W/2)x(W/2)  xself.nhidden1  #3x3x30=
        
        self.conv2 = nn.Conv2d(self.nhidden1,self.nhidden2,kernel_size=2,stride=1).to(device) # (W/2)-1 xself.nhidden2 #2x2x10=240
        
        self.conv3 = nn.Conv2d(self.nhidden2,self.nhidden3,kernel_size=2,stride=1).to(device) #(W/2)-2 xself.nhidden2 #1x1x80

        self.conv_out = nn.Conv2d(self.nhidden3, 16, 1).to(device) #16x(W/2)-1 


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))      
        x = F.softmax(self.conv_out(x),dim=1).squeeze()
        return x


    
class SeqNOCdefault(nn.Module):   #4000 epoch -> 0.82 loot
    def __init__(self,device=torch.device('cuda')):
        super().__init__()
        self.ctx_type=144 #6x6x4
        self.ksize = 6
        self.nhidden1 = 30
        self.nhidden2 = 30
        self.nhidden3 = 80
        self.conv1 = nn.Conv2d(4,self.nhidden1,kernel_size=2,stride=2).to(device) # (W/2)x(W/2)  xself.nhidden1  #3x3x30=
        
        self.conv2 = nn.Conv2d(self.nhidden1,self.nhidden2,kernel_size=2,stride=1).to(device) # (W/2)-1 xself.nhidden2 #2x2x10=240
        
        self.conv3 = nn.Conv2d(self.nhidden2,self.nhidden3,kernel_size=2,stride=1).to(device) #(W/2)-2 xself.nhidden2 #1x1x80


        self.conv_out = nn.Conv2d(self.nhidden3, 16, 1).to(device) #16x(W/2)-1 


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))      
        x = F.softmax(self.conv_out(x),dim=1).squeeze()
        return x


class SNHM(nn.Module):   
    def __init__(self,device=torch.device('cuda')):
        super().__init__()
        self.ctx_type=144 #6x6x4
        self.ksize = 6
        self.nhidden1 = 60
        self.nhidden2 = 60
        self.nhidden3 = 100
        self.conv1 = nn.Conv2d(4,self.nhidden1,kernel_size=2,stride=2).to(device) # (W/2)x(W/2)  xself.nhidden1  #3x3x30=
        
        self.conv2 = nn.Conv2d(self.nhidden1,self.nhidden2,kernel_size=2,stride=1).to(device) # (W/2)-1 xself.nhidden2 #2x2x10=240
        
        self.conv3 = nn.Conv2d(self.nhidden2,self.nhidden3,kernel_size=2,stride=1).to(device) #(W/2)-2 xself.nhidden2 #1x1x80


        self.conv_out = nn.Conv2d(self.nhidden3, 16, 1).to(device) #16x(W/2)-1 


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))      
        x = F.softmax(self.conv_out(x),dim=1).squeeze()
        return x





