#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:04:43 2022

@author: root
"""
import torch 

def tdilate_Loc(lrGT): 


    PatEl = torch.Tensor( [[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]]).int().to(torch.device('cuda'))
                       

    
    BlockiM = torch.zeros([0,3]).int().to(torch.device('cuda'))
    for iloc in range(8):    # % size(PatEl,1) = 8
        iv3 = PatEl[iloc,:]
        Blocki = lrGT*2+iv3
        # BlockiM = np.concatenate((BlockiM,Blocki),0) 
        BlockiM = torch.cat((BlockiM,Blocki)) 
    #LocM = np.unique(BlockiM,axis=0)
    return BlockiM



def torch_dilate_Loc(lrGT): 
    
    lrGT2 = lrGT*2
    a = torch.LongTensor( [[1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]]).to(lrGT.device)
    dlrGT = torch.cat((lrGT2,lrGT2+a[0,:],lrGT2+a[1,:],lrGT2+a[2,:],
                       lrGT2+a[3,:],lrGT2+a[4,:],lrGT2+a[5,:],lrGT2+a[6,:]))



    return dlrGT

def torch_Loc2LocM(Location):
    
    lrLoc = torch.unique(torch.div(Location,2,rounding_mode='trunc'),dim=0)
    
    return torch_dilate_Loc(lrLoc)

def torch_lowRes(Location):
    
    return torch.unique(torch.div(Location,2,rounding_mode='trunc'),dim=0)    
    
    
    
    
    
    