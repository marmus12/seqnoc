#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:10:52 2022

@author: root
"""
import torch
from pcloud_functs import pcread
from for_train_pt8 import get_uctxs_counts2,get_uctxs_counts5
import numpy as np
# from usefuls import dec2bin4
# nsymovs = np.zeros((16,),np.int8)
# for i in range(16):
#     nsymovs[i] = sum(dec2bin4(i,4))
def torch_criterion_mm(outputs,counts):
    return -torch.mean(torch.mean(counts*torch.log(outputs),1))

def torch_criterion(outputs,counts):
    return -torch.mean(torch.sum(counts*torch.log(outputs),1))

def mf_criterion(outputs,counts):
    return -torch.mean(torch.sum(counts*torch.log2(outputs),1))


def bpov_criterion(outputs,bcounts,device):
    s2np = torch.tensor([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4],dtype=torch.float32).unsqueeze(1).to(device)
    CL = - torch.sum(bcounts*torch.log2(outputs))
    npts = torch.sum(torch.matmul(bcounts,s2np))
    return CL/npts
    # return torch.mean((-torch.sum(counts*torch.log(outputs),1)/(1+torch.sum(bcounts*nsymovs,1)))*torch.sum(counts,1))
def bpov_criterion2(outputs,bcounts,npts,device):
    # s2np = torch.tensor([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4],dtype=torch.float32).unsqueeze(1).to(device)
    CL = - torch.sum(bcounts*torch.log2(outputs.squeeze()))
    return CL/npts

def bpov_criterion3(outputs10,bcounts10,outputs9,bcounts9,npts,device):
    # s2np = torch.tensor([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4],dtype=torch.float32).unsqueeze(1).to(device)
    CL = - torch.sum(bcounts10*torch.log2(outputs10)) - torch.sum(bcounts9*torch.log2(outputs9))
    return CL/npts



def compute_model_CL(state_dict,npbits):
    keys = state_dict.keys()
    nws = 0
    for key in keys:
        param = state_dict[key]
        nws+=np.prod(param.shape)   
        
    if npbits<32:
        mm_cost = 2*len(keys)*32 #cost of transmitting maxes and mins
    else:
        mm_cost=0
    mdl_CL = npbits*nws+mm_cost  

    return mdl_CL,nws

def writeCNNtobs(state_dict,bspath):
    keys = state_dict.keys()
    all_bytes = bytes()
    for key in keys:
        param = state_dict[key]
        parambytes = bytes(param.cpu().numpy())
        all_bytes = all_bytes+parambytes
    with open(bspath, "wb") as binary_file:
        binary_file.write(all_bytes)    

def readCNNfrombs(bspath,shapes=[]):
    with open(bspath, "rb") as binary_file:
        all_bytes = binary_file.read()  
    print(len(all_bytes))
    

def criterion_with_wreg(outputs,counts,lamda,wsum):
    return -torch.mean(torch.sum(counts*torch.log(outputs),1))+lamda*wsum


def collect_opt_data(filepaths,opt_frids,perms,device,ctx_type=144):
    counts_dtype = torch.int16
    counts = torch.zeros((0,16),dtype=counts_dtype).to(device)
    ctxs = torch.zeros((0,ctx_type),dtype=torch.bool).to(device)
    for i_frid,frid in enumerate(opt_frids):
        
            print(i_frid)
            ori_GT = pcread(filepaths[frid])#.astype('int')
            
            for perm in perms:
                print(str(perm))
                GT = ori_GT[:,perm]
                frctxs,frcounts = get_uctxs_counts2(GT,device,counts_dtype)
                
                ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
                counts = torch.cat((counts,frcounts),dim=0)#[all_ctxs ; ctxs];
                nall = ctxs.shape[0]
                
                # print('ifr/nfr:  ' + str(i_fr_tot) +'/' +str(nfr_tot))
                ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
            
                nu_ctxs = ctxs.shape[0]
                u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype).to(device)
            
                for i_actx in range(nall):
                    iuctx = uinvs[i_actx]
                    u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
                    if i_actx%100000==0:
                        print(str(i_actx) +'/' + str(nall ))    
                
            
                print('number of unique contexts up to now: '+str(nu_ctxs))
                
                counts = u_counts  
        

    
    
    ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
    counts=counts.type(torch.float32)
    return ctxs,counts,nu_ctxs


def collect_opt_data2(filepaths,opt_frids,perms,device,ctx_type=144,fourpass=False):
    
    ksize=np.sqrt(ctx_type//4).astype(int)
    if fourpass:
        ctx_dtype = torch.int8
    else:
        ctx_dtype = torch.bool
    counts_dtype = torch.int16
    counts = torch.zeros((0,16),dtype=counts_dtype).to(device)
    ctxs = torch.zeros((0,ctx_type),dtype=ctx_dtype).to(device)
    for i_frid,frid in enumerate(opt_frids):
        
            print(i_frid)
            ori_GT = pcread(filepaths[frid])#.astype('int')
            
            for perm in perms:
                print(str(perm))
                GT = ori_GT[:,perm]
                frctxs,frcounts = get_uctxs_counts2(GT,ksize,device,counts_dtype,fourpass=fourpass)
                
                ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
                counts = torch.cat((counts,frcounts),dim=0)#[all_ctxs ; ctxs];
                
                
                
    nall = ctxs.shape[0]
    ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
            


            
    nu_ctxs = ctxs.shape[0]
    u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype).to(device)

    for i_actx in range(nall):
        iuctx = uinvs[i_actx]
        u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
        if i_actx%100000==0:
            print(str(i_actx) +'/' + str(nall ))        
    
    

    
    
    u_ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
    u_counts=u_counts.type(torch.float32)
    return u_ctxs,u_counts,nu_ctxs


def collect_opt_data4(filepaths,opt_frids,perms,device,ctx_type=144,fourpass=False):
    
    ksize=np.sqrt(ctx_type//4).astype(int)
    if fourpass:
        ctx_dtype = torch.int8
    else:
        ctx_dtype = torch.bool
    counts_dtype = torch.int16
    counts = torch.zeros((0,16),dtype=counts_dtype).to(device)
    ctxs = torch.zeros((0,ctx_type),dtype=ctx_dtype).to(device)
    for i_frid,frid in enumerate(opt_frids):
        
            print(i_frid)
            ori_GT = pcread(filepaths[frid])#.astype('int')
            
            for perm in perms:
                print(str(perm))
                GT = ori_GT[:,perm]
                frctxs,frcounts = get_uctxs_counts2(GT,ksize,device,counts_dtype,fourpass=fourpass)
                
                ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
                counts = torch.cat((counts,frcounts),dim=0)#[all_ctxs ; ctxs];
                
                
                
    nall = ctxs.shape[0]
    ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
            


            
    nu_ctxs = ctxs.shape[0]
    u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype).to(device)
    u_counts[uinvs,:] = u_counts[uinvs,:]+counts 
    # for i_actx in range(nall):
    #     iuctx = uinvs[i_actx]
    #     u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
    #     if i_actx%100000==0:
    #         print(str(i_actx) +'/' + str(nall ))        

    u_ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
    u_counts=u_counts.type(torch.float32)
    return u_ctxs,u_counts,nu_ctxs

# def collect_opt_data3(filepaths,opt_frids,perms,device,ctx_type=144):
#     counts_dtype = torch.int16
#     counts = torch.zeros((0,16),dtype=counts_dtype).to(device)
#     ctxs = torch.zeros((0,ctx_type),dtype=torch.bool).to(device)
#     for i_frid,frid in enumerate(opt_frids):
        
#             print(i_frid)
#             ori_GT = pcread(filepaths[frid])#.astype('int')
            
#             for perm in perms:
#                 print(str(perm))
#                 GT = ori_GT[:,perm]
#                 frctxs,frcounts = get_uctxs_counts3(GT,device,counts_dtype)
                
#                 ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
#                 counts = torch.cat((counts,frcounts),dim=0)#[all_ctxs ; ctxs];
                
                
                
#     nall = ctxs.shape[0]
#     ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
            
#     # nu_ctxs = ctxs.shape[0]
#     # u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype).to(device)
    
#     # u_counts[uinvs,:] = u_counts[uinvs,:]+counts 
    

            
#     nu_ctxs = ctxs.shape[0]
#     u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype).to(device)

#     for i_actx in range(nall):
#         iuctx = uinvs[i_actx]
#         u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
#         if i_actx%100000==0:
#             print(str(i_actx) +'/' + str(nall ))        
    
    

    
    
#     u_ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
#     u_counts=u_counts.type(torch.float32)
#     return u_ctxs,u_counts,nu_ctxs


def collect_opt_data5(filepaths,opt_frids,perms,device,ctx_type=144,fourpass=False):
    
    ksize=np.sqrt(ctx_type//4).astype(int)
    if fourpass:
        ctx_dtype = torch.int8
    else:
        ctx_dtype = torch.bool
    counts_dtype = torch.int16
    counts = torch.zeros((0,16),dtype=counts_dtype).to(device)
    ctxs = torch.zeros((0,ctx_type),dtype=ctx_dtype).to(device)
    for i_frid,frid in enumerate(opt_frids):
        
            print(i_frid)
            ori_GT = pcread(filepaths[frid])#.astype('int')
            
            for perm in perms:
                print(str(perm))
                GT = ori_GT[:,perm]
                frctxs,frcounts = get_uctxs_counts5(GT,ksize,device,counts_dtype,fourpass=fourpass)
                
                ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
                counts = torch.cat((counts,frcounts),dim=0)#[all_ctxs ; ctxs];
                
                
                
    nall = ctxs.shape[0]
    ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
            

    nu_ctxs = ctxs.shape[0]
    print("number of unique ctxs: "+str(nu_ctxs))
    u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype).to(device)

    for i_actx in range(nall):
        iuctx = uinvs[i_actx]
        u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
        if i_actx%100000==0:
            print(str(i_actx) +'/' + str(nall ))        
    
   
    
    u_ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
    u_counts=u_counts.type(torch.float32)
    return u_ctxs,u_counts,nu_ctxs


def collect_opt_data6(filepaths,opt_frids,perms,device,ctx_type=144,fourpass=False):
    
    ksize=np.sqrt(ctx_type//4).astype(int)
    if fourpass:
        ctx_dtype = torch.int8
    else:
        ctx_dtype = torch.bool
    counts_dtype = torch.int16
    counts = torch.zeros((0,16),dtype=counts_dtype).to(device)
    ctxs = torch.zeros((0,ctx_type),dtype=ctx_dtype).to(device)
    for i_frid,frid in enumerate(opt_frids):
        
            print(i_frid)
            ori_GT = pcread(filepaths[frid])#.astype('int')
            
            for perm in perms:
                print(str(perm))
                GT = ori_GT[:,perm]
                frctxs,frcounts = get_uctxs_counts5(GT,ksize,device,counts_dtype,fourpass=fourpass)
                
                ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
                counts = torch.cat((counts,frcounts),dim=0)#[all_ctxs ; ctxs];
                
                
                
    # nall = ctxs.shape[0]
    ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
            

    nu_ctxs = ctxs.shape[0]
    print("number of unique ctxs: "+str(nu_ctxs))
    u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype).to(device)

    # for i_actx in range(nall):
    #     iuctx = uinvs[i_actx]
    #     u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
    #     if i_actx%100000==0:
    #         print(str(i_actx) +'/' + str(nall ))        
    
    
    for ikj in range(nu_ctxs):#ikj = 1:size(uCounts,1)
        u_counts[ikj,:] = u_counts[ikj,:] + torch.sum(counts[uinvs==ikj,:],0)

       
    
    u_ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
    u_counts=u_counts.type(torch.float32)
    return u_ctxs,u_counts,nu_ctxs


def collect_opt_data7(filepaths,opt_frids,perms,device,ctx_type=144,fourpass=False):
    
    ksize=np.sqrt(ctx_type//4).astype(int)
    if fourpass:
        ctx_dtype = torch.int8
    else:
        ctx_dtype = torch.bool
    counts_dtype = torch.int16
    counts = torch.zeros((0,16),dtype=counts_dtype).to(device)
    ctxs = torch.zeros((0,ctx_type),dtype=ctx_dtype).to(device)
    for i_frid,frid in enumerate(opt_frids):
        
            print(i_frid)
            ori_GT = pcread(filepaths[frid])#.astype('int')
            
            for perm in perms:
                print(str(perm))
                GT = ori_GT[:,perm]
                frctxs,frcounts = get_uctxs_counts5(GT,ksize,device,counts_dtype,fourpass=fourpass)
                
                ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
                counts = torch.cat((counts,frcounts),dim=0)#[all_ctxs ; ctxs];
                
                
                
    nall = ctxs.shape[0]
    ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
    
    counts = counts.to(torch.device('cpu'))
    uinvs = uinvs.to(torch.device('cpu'))
    nu_ctxs = ctxs.shape[0]
    print("number of unique ctxs: "+str(nu_ctxs))
    u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype)#.to(device)

    for i_actx in range(nall):
        iuctx = uinvs[i_actx]
        u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
        if i_actx%100000==0:
            print(str(i_actx) +'/' + str(nall ))        
    
   
    
    u_ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
    u_counts=u_counts.type(torch.float32).to(device)
    return u_ctxs,u_counts,nu_ctxs



def collect_opt_data8(filepaths,opt_frids,perms,device,ctx_type=144,fourpass=False,max_n_fr_ctxs=1000):
    
    ksize=np.sqrt(ctx_type//4).astype(int)
    if fourpass:
        ctx_dtype = torch.int8
    else:
        ctx_dtype = torch.bool
    counts_dtype = torch.int32
    counts = torch.zeros((0,16),dtype=counts_dtype)
    ctxs = torch.zeros((0,ctx_type),dtype=ctx_dtype).to(device)
    assert(len(perms)==1)
    perm = perms[0]
    for i_frid,frid in enumerate(opt_frids):
        
            print(i_frid)
            ori_GT = pcread(filepaths[frid])#.astype('int')
            
            # for perm in perms:
                # print(str(perm))
            GT = ori_GT[:,perm]
            frctxs,frcounts = get_uctxs_counts5(GT,ksize,device,counts_dtype,fourpass=fourpass)          
            frcounts =frcounts.to(torch.device('cpu'))
            nall = frctxs.shape[0]
            frctxs,uinvs = torch.unique(frctxs, return_inverse=True,dim=0)
            nu_fctxs = frctxs.shape[0]
            u_frcounts = torch.zeros((nu_fctxs,16),dtype=counts_dtype)
            
            for i_actx in range(nall):
                iuctx = uinvs[i_actx]
                u_frcounts[iuctx,:] = u_frcounts[iuctx,:]+frcounts[i_actx,:]
                if i_actx%100000==0:
                    print('for the frame: '+ str(i_actx) +'/' + str(nall))                
            
            # torch.argsort(torch.sum(u_frcounts,1)
            the_inds = torch.argsort(torch.sum(u_frcounts,1))[-max_n_fr_ctxs:]
            frctxs = frctxs[the_inds,:]
            u_frcounts = u_frcounts[the_inds,:]
            
            ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
            counts = torch.cat((counts,u_frcounts),dim=0)#[all_ctxs ; ctxs];                
                
    nall = ctxs.shape[0]
    ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
    
    #counts = counts.to(torch.device('cpu'))
    uinvs = uinvs.to(torch.device('cpu'))
    nu_ctxs = ctxs.shape[0]
    print("number of unique ctxs: "+str(nu_ctxs))
    u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype)#.to(device)

    for i_actx in range(nall):
        iuctx = uinvs[i_actx]
        u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
        if i_actx%100000==0:
            print(str(i_actx) +'/' + str(nall ))        
    
   
    
    u_ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
    u_counts=u_counts.type(torch.float32).to(device)
    return u_ctxs,u_counts,nu_ctxs


def collect_opt_data9(GTs,perms,device,ctx_type=144,fourpass=False):
    
    ksize=np.sqrt(ctx_type//4).astype(int)
    if fourpass:
        ctx_dtype = torch.int8
    else:
        ctx_dtype = torch.bool
    counts_dtype = torch.int16
    counts = torch.zeros((0,16),dtype=counts_dtype).to(device)
    ctxs = torch.zeros((0,ctx_type),dtype=ctx_dtype).to(device)
    for iGT,ori_GT in enumerate(GTs):
        
            print(iGT)
            # ori_GT = pcread(filepaths[frid])#.astype('int')
            if len(ori_GT)>0:
                for perm in perms:
                    print(str(perm))
                    GT = ori_GT[:,perm]
                    frctxs,frcounts = get_uctxs_counts5(GT,ksize,device,counts_dtype,fourpass=fourpass)
                    
                    ctxs = torch.cat((ctxs,frctxs),dim=0)#[all_ctxs ; ctxs];
                    counts = torch.cat((counts,frcounts),dim=0)#[all_ctxs ; ctxs];
                
                
                
    nall = ctxs.shape[0]
    ctxs,uinvs = torch.unique(ctxs, return_inverse=True,dim=0)
    
    counts = counts.to(torch.device('cpu'))
    uinvs = uinvs.to(torch.device('cpu'))
    nu_ctxs = ctxs.shape[0]
    print("number of unique ctxs: "+str(nu_ctxs))
    u_counts = torch.zeros((nu_ctxs,16),dtype=counts_dtype)#.to(device)

    for i_actx in range(nall):
        iuctx = uinvs[i_actx]
        u_counts[iuctx,:] = u_counts[iuctx,:]+counts[i_actx,:]
        if i_actx%100000==0:
            print(str(i_actx) +'/' + str(nall ))        
    
   
    
    u_ctxs = ctxs.reshape((nu_ctxs,4,6,6)).type(torch.float32)
    u_counts=u_counts.type(torch.float32).to(device)
    return u_ctxs,u_counts,nu_ctxs






