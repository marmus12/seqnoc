#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:36:29 2021

@author: root
"""
import torch
import torchac
from array import array
import numpy as np
import sys
from pcloud_functs import pcshow,pcread,lowerResolution,inds2vol,vol2inds,dilate_Loc
sys.path.append('/home/emre/Documents/kodlar/Reference-arithmetic-coding-master/python/')
from usefuls import in1d_index,plt_imshow,write_ints,read_ints,write_bits,read_bits,dec2bin4,dec2bin2_fast,bin2dec2,ints2bs,bs2ints


import globz
from time import time
from torch_functs import torch_Loc2LocM


def get_temps_dests(LocM,ksize,device):

    # LocM = torch.LongTensor(LocM).to(device)   

    theshift = globz.theshift

    LocM = LocM+theshift


    maxX,maxY,maxZ = torch.max(LocM,axis=0).values.cpu().numpy()+theshift
        
    VolM = torch.zeros((maxY,maxX,maxZ),dtype=torch.int8).to(device)
    # try:
    VolM[LocM[:,1],LocM[:,0],LocM[:,2]]=1
    


    globz.Loc = globz.Loc+theshift
    
    VolL = torch.zeros((maxY,maxX,maxZ)).to(device)
    VolL[globz.Loc[:,1],globz.Loc[:,0],globz.Loc[:,2]]=1



    # dssx,dssz=(maxX-2)//2-1,(maxZ-2)//2-1
    dssx,dssz = torch.div(maxX-2,2,rounding_mode='trunc'),torch.div(maxZ-2,2,rounding_mode='trunc')
    # dsBWTrueM = torch.zeros((dssx,dssz)).bool().to(device)
    
    
    maxnT = 1000000
    Temps = torch.zeros((maxnT,4,36),dtype=torch.int8).to(device)
    Dess = torch.zeros((maxnT,),dtype=torch.int8).to(device)
    b2dmult = torch.Tensor([[[[8,4],[2,1]]]]).to(device)

    iT = 0
    for icPC in range(theshift,maxY-theshift):
        
        if iT <(maxnT-10000):
            
            if icPC%100==0:
                print('icPC:' + str(icPC))
    
    
            # smask = VolM[icPC,2::2,2::2].bool()
            # dsBWTrueM =smask[0:dssx,0:dssz]
            # dsBWTrueM1  = VolM[icPC,0::2,0::2].bool()
            dsBWTrueM  = VolM[icPC,2:(2*dssx):2,2:(2*dssz):2].bool()
            
            # if not ENC:
            nzdsBM = torch.nonzero(dsBWTrueM)#*2
            
            ncand = nzdsBM.shape[0]
            if ncand>0:              
                

                BWT = VolL[icPC:(icPC+1),2:-2,2:-2].unsqueeze(0)
                symbs_enc = torch.conv2d(BWT,b2dmult,stride=[2,2])#.squeeze(0).squeeze(0)

                Dess[iT:(iT+ncand)] = symbs_enc.masked_select(dsBWTrueM)
                

                unfolded = VolM[(icPC-2):(icPC+2),:,:].unfold(1,6,2).unfold(2,6,2)
                unf2=unfolded.masked_select(dsBWTrueM.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                Temps[iT:(iT+ncand),:] = unf2.reshape([4,ncand,36]).permute((1,0,2))#.transpose(0,1)
                
                iT+=ncand

                    
            VolM[icPC,:,:] = VolL[icPC,:,:]
            
        
    return Temps[0:iT,:],Dess[0:iT]

def get_temps_dests4p2(LocM,device):



    theshift = globz.theshift

    LocM = LocM+theshift


    maxX,maxY,maxZ = torch.max(LocM,axis=0).values.cpu().numpy()+theshift
        
    VolM = torch.zeros((maxY,maxX,maxZ),dtype=torch.int8).to(device)
    # try:
    VolM[LocM[:,1],LocM[:,0],LocM[:,2]]=2


    globz.Loc = globz.Loc+theshift
    
    VolL = torch.zeros((maxY,maxX,maxZ)).to(device)
    VolL[globz.Loc[:,1],globz.Loc[:,0],globz.Loc[:,2]]=1



    # dssx,dssz=(maxX-2)//2-1,(maxZ-2)//2-1
    dssx,dssz = torch.div(maxX-2,2,rounding_mode='trunc'),torch.div(maxZ-2,2,rounding_mode='trunc')
    # dsBWTrueM = torch.zeros((dssx,dssz)).bool().to(device)
    mask0 = torch.zeros((dssx-1,dssz-1)).bool().to(device)
    mask1 = torch.zeros((dssx-1,dssz-1)).bool().to(device)
    mask2 = torch.zeros((dssx-1,dssz-1)).bool().to(device)
    mask3 = torch.zeros((dssx-1,dssz-1)).bool().to(device)
    
    maxnT = 1000000
    Temps = torch.zeros((maxnT,4,36),dtype=torch.int8).to(device)
    Dess = torch.zeros((maxnT,),dtype=torch.int8).to(device)
    b2dmult = torch.Tensor([[[[8,4],[2,1]]]]).to(device)
    
    
    szx = 4*(maxX//4)
    szz = 4*(maxZ//4)

    
    Tmask0 = torch.zeros([maxX,maxZ]).bool().to(device)
    patch0 = torch.zeros([4,4]).bool()
    patch0[0:2,0:2] = True
    
    Tmask0[2:(2+szx),2:(2+szz)] = patch0.repeat((maxX//4,maxZ//4))[0:(maxX-2),0:(maxZ-2)]
    # inds0x,inds0z = torch.where(Tmask0)
    ######################################
    Tmask1 = torch.zeros([maxX,maxZ]).bool().to(device)    
    patch1 = torch.zeros([4,4]).bool()
    patch1[2:,2:] = True
    
    Tmask1[2:(2+szx),2:(2+szz)] = patch1.repeat((maxX//4,maxZ//4))[0:(maxX-2),0:(maxZ-2)]  
    # inds1x,inds1z = torch.where(Tmask1)    
################################################3    
    Tmask2 = torch.zeros([maxX,maxZ]).bool().to(device)    
    patch2 = torch.zeros([4,4]).bool()
    patch2[2:,0:2] = True
    
    Tmask2[2:(2+szx),2:(2+szz)] = patch2.repeat((maxX//4,maxZ//4))[0:(maxX-2),0:(maxZ-2)]  
    # inds2x,inds2z = torch.where(Tmask2)
################################################3    

     
    iT = 0
    for icPC in range(theshift,maxY-theshift):
        
        if iT <(maxnT-10000):
            
            if icPC%100==0:
                print('icPC:' + str(icPC))
    

            dsBWTrueM = (VolM[icPC,2:(2*dssx):2,2:(2*dssz):2]/2).bool()

            nzdsBM = torch.nonzero(dsBWTrueM)#*2
            
            
            ncand = nzdsBM.shape[0]
            if ncand>0:              
                VolM4 = VolM[(icPC-2):(icPC+2),:,:]
                
                mask0[0::2,0::2] = dsBWTrueM[0::2,0::2]
                mask1[1::2,1::2]  = dsBWTrueM[1::2,1::2]     
                mask2[1::2,0::2] = dsBWTrueM[1::2,0::2]                
                mask3[0::2,1::2] = dsBWTrueM[0::2,1::2]

                BWT0 = VolL[icPC:(icPC+1),:,:]
                BWT = BWT0[:,2:-2,2:-2].unsqueeze(0)
                BWT0 = BWT0.squeeze()

                
                symbs_enc = torch.conv2d(BWT,b2dmult,stride=[2,2])


                
                ncand0 = torch.nonzero(mask0).shape[0]
                ncand1 = torch.nonzero(mask1).shape[0]
                ncand2 = torch.nonzero(mask2).shape[0]
                ncand3 = torch.nonzero(mask3).shape[0]
                
                if ncand0>0:
                    ######COLLECT 0th Pass: ########################
                    unfolded = VolM4.unfold(1,6,2).unfold(2,6,2)
                    
                    unf=unfolded.masked_select(mask0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                    
    
                    Temps[iT:(iT+ncand0),:] = unf.reshape([4,ncand0,36]).permute((1,0,2))#.transpose(0,1)
                    
                    Dess[iT:(iT+ncand0)] = symbs_enc.masked_select(mask0)
                    iT+=ncand0
                
                VolM4[2,:,:] =  VolM4[2,:,:] + ((BWT0)*2+1-2)*Tmask0*(VolM4[2,:,:]/2) #True zero:1, True one:3
                
                if ncand1>0:
                    ######COLLECT 1sd Pass: ########################
                    # VolM4[2,2:-2:2,2:-2:2] = BWT[0,0,0::2,0::2]*2+1 #True zero:1, True one:3

                    unfolded = VolM4.unfold(1,6,2).unfold(2,6,2)
                    
                    unf=unfolded.masked_select(mask1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                    
                    Temps[iT:(iT+ncand1),:] = unf.reshape([4,ncand1,36]).permute((1,0,2))#.transpose(0,1)
                    
                    Dess[iT:(iT+ncand1)] = symbs_enc.masked_select(mask1)
                    
                    iT+=ncand1
                    
                VolM4[2,:,:] =  VolM4[2,:,:] + ((BWT0)*2-1)*Tmask1*(VolM4[2,:,:]/2)          
                
                if ncand2>0:
                    ######COLLECT 2nd Pass: ########################

                    
                    unfolded = VolM4.unfold(1,6,2).unfold(2,6,2)
                    
                    unf=unfolded.masked_select(mask2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                    
                    Temps[iT:(iT+ncand2),:] = unf.reshape([4,ncand2,36]).permute((1,0,2))
                    
                    Dess[iT:(iT+ncand2)] = symbs_enc.masked_select(mask2)
                    iT+=ncand2
                    
                VolM4[2,:,:] =  VolM4[2,:,:] + ((BWT0)*2-1)*Tmask2*(VolM4[2,:,:]/2)    #True zero:1, True one:3  
                
                if ncand3>0:
                    ######COLLECT 3rd Pass: ########################

                    
                    unfolded = VolM4.unfold(1,6,2).unfold(2,6,2)
                    
                    unf=unfolded.masked_select(mask3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                    
                    Temps[iT:(iT+ncand3),:] = unf.reshape([4,ncand3,36]).permute((1,0,2))
                    
                    Dess[iT:(iT+ncand3)] = symbs_enc.masked_select(mask3)
                    iT+=ncand3               
                
                    
            VolM[icPC,:,:] = VolL[icPC,:,:]
            
        
    return Temps[0:iT,:],Dess[0:iT]


def get_temps_dests4p(LocM,ksize,device):

    delta = (ksize-2)//2

    theshift = globz.theshift

    LocM = LocM+theshift


    maxX,maxY,maxZ = torch.max(LocM,axis=0).values.cpu().numpy()+theshift
        
    VolM = torch.zeros((maxY,maxX,maxZ),dtype=torch.int8).to(device)
    # try:
    VolM[LocM[:,1],LocM[:,0],LocM[:,2]]=2


    globz.Loc = globz.Loc+theshift
    
    VolL = torch.zeros((maxY,maxX,maxZ)).to(device)
    VolL[globz.Loc[:,1],globz.Loc[:,0],globz.Loc[:,2]]=1



    # dssx,dssz=(maxX-2)//2-1,(maxZ-2)//2-1
    dssx,dssz = torch.div(maxX-delta,2,rounding_mode='trunc'),torch.div(maxZ-delta,2,rounding_mode='trunc')
    # dsBWTrueM = torch.zeros((dssx,dssz)).bool().to(device)
    mask0 = torch.zeros((dssx-delta//2,dssz-delta//2)).bool().to(device)
    mask1 = torch.zeros((dssx-delta//2,dssz-delta//2)).bool().to(device)
    mask2 = torch.zeros((dssx-delta//2,dssz-delta//2)).bool().to(device)
    mask3 = torch.zeros((dssx-delta//2,dssz-delta//2)).bool().to(device)
    
    maxnT = 1000000
    Temps = torch.zeros((maxnT,4,ksize**2),dtype=torch.int8).to(device)
    Dess = torch.zeros((maxnT,),dtype=torch.int8).to(device)
    b2dmult = torch.Tensor([[[[8,4],[2,1]]]]).to(device)
    
    
    szx = 4*(maxX//4)
    szz = 4*(maxZ//4)

    
    Tmask0 = torch.zeros([maxX,maxZ]).bool().to(device)
    patch0 = torch.zeros([4,4]).bool()
    patch0[0:2,0:2] = True
    
    Tmask0[delta:(delta+szx),delta:(delta+szz)] = patch0.repeat((maxX//4,maxZ//4))[0:(maxX-delta),0:(maxZ-delta)]
    # inds0x,inds0z = torch.where(Tmask0)
    ######################################
    Tmask1 = torch.zeros([maxX,maxZ]).bool().to(device)    
    patch1 = torch.zeros([4,4]).bool()
    patch1[0:2,2:] = True
    
    Tmask1[delta:(delta+szx),delta:(delta+szz)] = patch1.repeat((maxX//4,maxZ//4))[0:(maxX-delta),0:(maxZ-delta)]  
    # inds1x,inds1z = torch.where(Tmask1)    
################################################3    
    Tmask2 = torch.zeros([maxX,maxZ]).bool().to(device)    
    patch2 = torch.zeros([4,4]).bool()
    patch2[2:,0:2] = True
    
    Tmask2[delta:(delta+szx),delta:(delta+szz)] = patch2.repeat((maxX//4,maxZ//4))[0:(maxX-delta),0:(maxZ-delta)]  
    # inds2x,inds2z = torch.where(Tmask2)
################################################3    
    # Tmask3 = torch.zeros([maxX,maxZ]).bool()    
    # patch3 = torch.zeros([4,4]).bool()
    # patch3[2:,2:] = True
    
    # Tmask3[2:(2+szx),2:(2+szz)] = patch3.repeat((maxX//4,maxZ//4))    
    # inds3x,inds3z = torch.where(Tmask3)     
     
    iT = 0
    for icPC in range(theshift,maxY-theshift):
        
        if iT <(maxnT-10000):
            
            if icPC%100==0:
                print('icPC:' + str(icPC))
    

            dsBWTrueM = (VolM[icPC,delta:(2*dssx):2,delta:(2*dssz):2]/2).bool()

            nzdsBM = torch.nonzero(dsBWTrueM)#*2
            
            
            ncand = nzdsBM.shape[0]
            if ncand>0:              
                VolM4 = VolM[(icPC-2):(icPC+2),:,:]
                
                mask0[0::2,0::2] = dsBWTrueM[0::2,0::2]
                mask1[0::2,1::2] = dsBWTrueM[0::2,1::2]
                mask2[1::2,0::2] = dsBWTrueM[1::2,0::2]                
                mask3[1::2,1::2]  = dsBWTrueM[1::2,1::2]                

                BWT0 = VolL[icPC:(icPC+1),:,:]
                BWT = BWT0[:,delta:-delta,delta:-delta].unsqueeze(0)
                BWT0 = BWT0.squeeze()

                
                symbs_enc = torch.conv2d(BWT,b2dmult,stride=[2,2])


                
                ncand0 = torch.nonzero(mask0).shape[0]
                ncand1 = torch.nonzero(mask1).shape[0]
                ncand2 = torch.nonzero(mask2).shape[0]
                ncand3 = torch.nonzero(mask3).shape[0]
                
                if ncand0>0:
                    ######COLLECT 0th Pass: ########################
                    unfolded = VolM4.unfold(1,ksize,2).unfold(2,ksize,2)
                    
                    unf=unfolded.masked_select(mask0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                    
    
                    Temps[iT:(iT+ncand0),:] = unf.reshape([4,ncand0,ksize**2]).permute((1,0,2))#.transpose(0,1)
                    
                    Dess[iT:(iT+ncand0)] = symbs_enc.masked_select(mask0)
                    iT+=ncand0
                
                VolM4[2,:,:] =  VolM4[2,:,:] + ((BWT0)*2+1-2)*Tmask0*(VolM4[2,:,:]/2) #True zero:1, True one:3
                
                if ncand1>0:
                    ######COLLECT 1sd Pass: ########################
                    # VolM4[2,2:-2:2,2:-2:2] = BWT[0,0,0::2,0::2]*2+1 #True zero:1, True one:3

                    unfolded = VolM4.unfold(1,ksize,2).unfold(2,ksize,2)
                    
                    unf=unfolded.masked_select(mask1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                    
                    Temps[iT:(iT+ncand1),:] = unf.reshape([4,ncand1,ksize**2]).permute((1,0,2))#.transpose(0,1)
                    
                    Dess[iT:(iT+ncand1)] = symbs_enc.masked_select(mask1)
                    
                    iT+=ncand1
                    
                VolM4[2,:,:] =  VolM4[2,:,:] + ((BWT0)*2-1)*Tmask1*(VolM4[2,:,:]/2)          
                
                if ncand2>0:
                    ######COLLECT 2nd Pass: ########################

                    
                    unfolded = VolM4.unfold(1,ksize,2).unfold(2,ksize,2)
                    
                    unf=unfolded.masked_select(mask2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                    
                    Temps[iT:(iT+ncand2),:] = unf.reshape([4,ncand2,ksize**2]).permute((1,0,2))
                    
                    Dess[iT:(iT+ncand2)] = symbs_enc.masked_select(mask2)
                    iT+=ncand2
                    
                VolM4[2,:,:] =  VolM4[2,:,:] + ((BWT0)*2-1)*Tmask2*(VolM4[2,:,:]/2)    #True zero:1, True one:3  
                
                if ncand3>0:
                    ######COLLECT 3rd Pass: ########################

                    
                    unfolded = VolM4.unfold(1,ksize,2).unfold(2,ksize,2)
                    
                    unf=unfolded.masked_select(mask3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                    
                    Temps[iT:(iT+ncand3),:] = unf.reshape([4,ncand3,ksize**2]).permute((1,0,2))
                    
                    Dess[iT:(iT+ncand3)] = symbs_enc.masked_select(mask3)
                    iT+=ncand3               
                
                    
            VolM[icPC,:,:] = VolL[icPC,:,:]
            
        
    return Temps[0:iT,:],Dess[0:iT]


    
        
def get_uctxs_counts2(Location,ksize,device,counts_dtype,fourpass = False):
        if fourpass:
            GTD_funct = get_temps_dests4p
        else:
            GTD_funct = get_temps_dests
            
        globz.Loc = torch.LongTensor(Location).to(device)    
            
        # LocM = dilate_Loc(lowerResolution(Location))
        LocM = torch_Loc2LocM(globz.Loc)

        # LocM = torch.LongTensor(LocM).to(device)   

        Temps,syms = GTD_funct(LocM,ksize,device)
        nTemps  =Temps.shape[0]
        
        fTemps = Temps.reshape((nTemps,-1))
    
        Ts = torch.cat((fTemps,syms.unsqueeze(1)),1)
        uTs,counts1 = torch.unique(Ts,return_counts=True,dim=0)
        
        
        uctxs,ic1=torch.unique(uTs[:,0:-1],return_inverse=True,dim=0)
        
        nTs = uTs.shape[0]
        nuctxs = uctxs.shape[0]
        counts = torch.zeros((nuctxs,16),dtype=counts_dtype).to(device)
    
        # sfsdf=5
        
        for its in range(nTs):
            sym = uTs[its,-1]
            counts[ic1[its],sym]+=counts1[its] #.item()

    
        
        return uctxs,counts
    

##################################################################






def get_uctxs_counts3(Location,device,counts_dtype): #WORKS BUT SLOWER!


        LocM = dilate_Loc(lowerResolution(Location))
 

        globz.Loc = torch.LongTensor(Location).to(device)

        Temps,syms = get_temps_dests(LocM,device)
        
        onehot_syms = torch.nn.functional.one_hot(syms.type(torch.long),16) #nTempsx16
        nTemps  =Temps.shape[0]
        
        fTemps = Temps.reshape((nTemps,-1))
    
        uctxs,ic1=torch.unique(fTemps,return_inverse=True,dim=0)
    
        

        nuctxs = uctxs.shape[0]
        counts = torch.zeros((nuctxs,16),dtype=counts_dtype).to(device)
        
        for iuc in range(nuctxs):
            counts[iuc,:] = torch.sum(onehot_syms[ic1==iuc,:],0)
            

    
        
        return uctxs,counts
    

def get_uctxs_counts4(Location,ksize,device,counts_dtype,fourpass = False):
        if fourpass:
            GTD_funct = get_temps_dests4p
        else:
            GTD_funct = get_temps_dests
            
        globz.Loc = torch.LongTensor(Location).to(device)    
            
        # LocM = dilate_Loc(lowerResolution(Location))
        LocM = torch_Loc2LocM(globz.Loc)

        # LocM = torch.LongTensor(LocM).to(device)   

        Temps,syms = GTD_funct(LocM,ksize,device)
        nTemps  =Temps.shape[0]
        
        fTemps = Temps.reshape((nTemps,-1))
    
        Ts = torch.cat((fTemps,syms.unsqueeze(1)),1)
        uTs,counts1 = torch.unique(Ts,return_counts=True,dim=0)
        
        
        uctxs,ic1=torch.unique(uTs[:,0:-1],return_inverse=True,dim=0)
        
        nTs = uTs.shape[0]
        nuctxs = uctxs.shape[0]
        counts = torch.zeros((nuctxs,16),dtype=counts_dtype).to(device)
    
        # sfsdf=5
        
        for its in range(nTs):
            sym = uTs[its,-1]
            counts[ic1[its],sym]+=counts1[its] #.item()
        
        # counts[ic1] = counts1
    
        
        return uctxs,counts


def get_uctxs_counts5(Location,ksize,device,counts_dtype,fourpass = False):
        if fourpass:
            GTD_funct = get_temps_dests4p
        else:
            GTD_funct = get_temps_dests
            
        globz.Loc = torch.LongTensor(Location).to(device)    
            
        # LocM = dilate_Loc(lowerResolution(Location))
        LocM = torch_Loc2LocM(globz.Loc)

        # LocM = torch.LongTensor(LocM).to(device)   

        Temps,syms = GTD_funct(LocM,ksize,device)
        
        onehot_syms = torch.nn.functional.one_hot(syms.type(torch.long),16) #nTempsx16
        nTemps  =Temps.shape[0]
        
        fTemps = Temps.reshape((nTemps,-1))
    

        
        return fTemps,onehot_syms










