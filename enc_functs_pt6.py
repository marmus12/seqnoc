#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:36:29 2021

@author: root
"""
import torch
from torch_functs import torch_dilate_Loc
import torchac
from array import array
import numpy as np
import sys
from pcloud_functs import pcshow,pcread,lowerResolution,inds2vol,vol2inds,dilate_Loc
sys.path.append('/home/emre/Documents/kodlar/Reference-arithmetic-coding-master/python/')
from usefuls import in1d_index,plt_imshow,write_ints,read_ints,write_bits,read_bits,dec2bin4,dec2bin2_fast,bin2dec2,ints2bs,bs2ints
# import arithmeticcoding as arc
# import tensorflow.compat.v1 as tf1
import globz
from time import time

# from dec2bin import dec2bin
from runlength import RLED
from GR import GR_ENCODE_DECODE

import MinkowskiEngine as ME


# st2 = time()
# globz.t2 += time()-st2       
def get_temps_dests2(ENC=True,nn_model ='dec',bs_dir=None,level=0,ori_level=0,dSSLs=0,LocM=0,lbss_in=0,ac_stream_in=0,device=0):

    

    cpud = torch.device('cpu')
    iBBr = 0
    theshift = globz.theshift

    LocM = LocM+theshift#torch.LongTensor([theshift ,theshift ,theshift ])

    # maxX = maxesL[0]  
    # maxY = maxesL[1]
    # maxZ = maxesL[2]maxes
    maxX,maxY,maxZ = torch.max(LocM,axis=0).values+theshift#+[theshift ,theshift ,theshift ]
        
    VolM = torch.zeros((maxY,maxX,maxZ)).to(device)
    # try:
    VolM[LocM[:,1],LocM[:,0],LocM[:,2]]=1
    




    # %% Find sections in Loc
    lSSL = maxY.item()#+1
 
    ac_stream = bytes()
    if ENC:
        
        
        globz.Loc = globz.Loc+theshift#torch.LongTensor([theshift ,theshift ,theshift ]).to(device)
        
        VolL = torch.zeros((maxY,maxX,maxZ)).to(device)
        VolL[globz.Loc[:,1],globz.Loc[:,0],globz.Loc[:,2]]=1


        SSL2 = np.zeros((lSSL,),'int8')
        SSL2[globz.Loc[:,1].cpu()]=1

        
        if level==ori_level :
            # st5=time()
            ssbits = ''
            for ssbit in SSL2:
                ssbits=ssbits+str(ssbit)
            # globz.t5+=time()-st5
            

            
            RLED(ssbits,lSSL,lSSL,1,bs_dir+'rSSL.dat')
            
    else:    
        

        
        SSL2 = dSSLs[level,:]
        
        
        
    ncPC = np.max(np.where(SSL2)[0])

    # %% Find sections in LocM


    if not ENC:        
        lLocM = LocM.shape[0] 
        globz.Loc = torch.zeros((lLocM ,3),dtype=torch.int16).to(device)
    # shiftsxz = np.array([[0,0,1,1],[0,1,0,1]],'int')
    
    # dssx,dssz=(maxX-2)//2-1,(maxZ-2)//2-1
    dssx,dssz = torch.div(maxX-2,2,rounding_mode='trunc'),torch.div(maxZ-2,2,rounding_mode='trunc')
    # dsBWTrueM = torch.zeros((dssx,dssz)).bool().to(device)
    TempBWs = torch.zeros((1,4,maxX,maxZ)).to(device)

    # shiftsz = (0,1,0,1) 
    b2dmult = torch.Tensor([[[[8,4],[2,1]]]]).to(device)
    lbss = np.zeros((ncPC+1,),'int16')
    total_lbs = 0
    Wp2c = (1-torch.tril(torch.ones((16,17)))).to(device)
    # Wp2c[:,-1]=0
    for icPC in range(theshift,ncPC+1):
        
        if icPC%300==0:
            print('icPC:' + str(icPC))


            
        if SSL2[icPC]:
            
     
            if ENC:
                
                TempBWs[0,0:2,:,:] = VolL[(icPC-2):(icPC),:,:] #BWTrue2
                


            TempBWs[0,2:4,:,:] = VolM[icPC:(icPC+2),:,:]
            

            dsBWTrueM  = VolM[icPC,2:(2*dssx):2,2:(2*dssz):2].bool()
            # dsBWTrueM =smask[0:dssx,0:dssz]            
            # if not ENC:
            nzdsBM = torch.nonzero(dsBWTrueM)

            lSLocM = nzdsBM.shape[0]#torch.sum(dsBWTrueM)#SLocM.shape[0]#len(SLocMx)#.shape[0]
            #if lSLocM>0: 
            

            nn_output = nn_model.forward(TempBWs)#.to(cpud)
            # st4 = time()
            output_pdf = nn_output.masked_select(dsBWTrueM).reshape([16,lSLocM]).transpose(0,1)
            # globz.t4+= time()-st4

            good_cdf = torch.clamp(torch.matmul(output_pdf,Wp2c),0,1).to(cpud)
            # st3 = time()
            # good_cdf = good_cdf.to(cpud)
            # globz.t3+= time()-st3
        ##########2nd loop##########################################
            # iT2=0
            if ENC:
                
                # BWTrue = VolL[icPC:(icPC+1),:,:].unsqueeze(0)


                symbs_enc = torch.conv2d(VolL[icPC:(icPC+1),2:-2,2:-2].unsqueeze(0),b2dmult,stride=[2,2])#.squeeze(0).squeeze(0)

                tsymbs = symbs_enc.masked_select(dsBWTrueM).type(torch.int16).to(cpud)

                byte_stream = torchac.encode_float_cdf(good_cdf, tsymbs, check_input_bounds=True)
                # except:
                #      asagfa=5
                # lbs = len(byte_stream)
                lbss[icPC] = len(byte_stream)                   
                ac_stream = ac_stream+byte_stream
                
            else:

                lbs = lbss_in[icPC]
                byte_stream = ac_stream_in[total_lbs:(total_lbs+lbs)]
                dec_syms = torchac.decode_float_cdf(good_cdf,byte_stream)#.tolist()
                # print(dec_syms)
                total_lbs += lbs

                
                
                block_sym = torch.zeros((lSLocM,2,2),dtype=torch.bool)#.to(device)
                block_sym[:,0,0] = torch.div(dec_syms,8,rounding_mode='trunc')
                block_sym[:,0,1] =  torch.div((dec_syms%8),4,rounding_mode='trunc')
                block_sym[:,1,0] = torch.div((dec_syms%4),2,rounding_mode='trunc')
                block_sym[:,1,1] = dec_syms%2
                ixs = (nzdsBM*2+2)[:,0]
                izs = (nzdsBM*2+2)[:,1]
                # shxs = torch.cat((ixs,ixs,ixs+1,ixs+1)) #4xnsyms
                # shzs = torch.cat((izs,izs+1,izs,izs+1)) #4xnsyms
                shxs1 = ixs[block_sym[:,0,0]]      
                shzs1 = izs[block_sym[:,0,0]]  
                shxs2 = ixs[block_sym[:,0,1]]      
                shzs2 = izs[block_sym[:,0,1]]  
                shxs3 = ixs[block_sym[:,1,0]]      
                shzs3 = izs[block_sym[:,1,0]]  
                shxs4 = ixs[block_sym[:,1,1]]      
                shzs4 = izs[block_sym[:,1,1]]  
                shxs = torch.cat((shxs1 ,shxs2 ,shxs3+1,shxs4+1)) #4xnsyms
                shzs = torch.cat((shzs1 ,shzs2+1,shzs3,shzs4+1)) #4xnsyms
                TempBWs[0,2,:,:] = 0
                TempBWs[0,2,shxs,shzs] = 1
                npts = shxs.shape[0]
                # for ip in range(npts):
                # new_pts = icPC*torch.ones((npts,3),dtype=torch.int).to(device)
                globz.Loc[iBBr:(iBBr+npts),0] = shxs
                globz.Loc[iBBr:(iBBr+npts),1] = icPC
                globz.Loc[iBBr:(iBBr+npts),2] = shzs
                iBBr +=npts



            
        if not ENC:
            TempBWs[0,0,:] = TempBWs[0,1,:]
            TempBWs[0,1,:] = TempBWs[0,2,:]
            
            


    if ENC:    
        dec_Loc = 0
    if not(ENC) :#and not(for_train):
        dec_Loc =  (globz.Loc[0:iBBr,:]-theshift)#.cpu().numpy()


    return dec_Loc,ac_stream,lbss
    
        


    

    
def ENCODE_DECODE(ENC,bs_dir,nn_model,ori_level=0,GT=0,device = torch.device('cuda')):
    

    # cpud = torch.device('cpu')
    
    # tGT = torch.Tensor(GT).to(device)
    
    start = time()      

    globz.theshift=16
    nintbits = ori_level*np.ones((6,),int)
    lrGTs = dict()
    if ENC:# or debug_dec:
        tGT = torch.LongTensor(GT).to(device)
        minsG = np.min(GT ,0)
        maxesG = np.max(GT ,0)
        minmaxesG = np.concatenate((minsG,maxesG))
        
        # write_ints(minmaxesG,nintbits,bs_dir+'maxes_mins.dat')    
        sibs = ints2bs(minmaxesG,nintbits)
        
        lrGTs[ori_level] = tGT
        lrGT = tGT#np.copy(GT)
        for il in range(ori_level-2):
            lrGT = torch.unique(torch.div(lrGT,2,rounding_mode='trunc'),dim=0)#lowerResolution(lrGT) #lrGT//2#
            lrGTs[ori_level-il-1] = lrGT
            
        lowest_bs = inds2vol(lrGTs[2].cpu().numpy(),[4,4,4]).flatten().astype(int)
        lowest_str = ''
        for ibit in range(64):
            lowest_str = lowest_str+str(lowest_bs[ibit])
            
        #write_bits(lowest_str+'1',bs_dir+'lowest.dat')
        sibs = sibs+lowest_str+'1'
        write_bits(sibs,bs_dir+'side_info.bs')
        dSSLs = 0
    if not ENC:


        sibs = read_bits(bs_dir+'side_info.bs')
        minmaxesG = bs2ints(sibs[0:np.sum(nintbits)],nintbits)
        lowest_str = sibs[np.sum(nintbits):-1]
        
        lowest_bs = np.zeros([64,],int)
        for ibit in range(64):
            lowest_bs[ibit] = int(lowest_str[ibit])
        vol = lowest_bs.reshape([4,4,4])
        lrGTs[2] = torch.LongTensor(vol2inds(vol)).to(device)

     
        
        lrmm = np.copy(minmaxesG[np.newaxis,:])
        lrmms=np.zeros((ori_level+1,6),int)
        lrmms[ori_level] = lrmm
        for il in range(ori_level-2):
            lrmm = lrmm//2#lowerResolution(lrmm)
            lrmms[ori_level-il-1,:] = lrmm
            
        mins11 = lrmms[ori_level,1]
        maxes11 = lrmms[ori_level,4]        
        lSSL = maxes11+2*globz.theshift#+32+10#+40
                
       ##get dssls     
        dSSLs  = np.zeros((ori_level+1,4500),int)

        # ssbits =''# 32*'0'
        ssbits = RLED('',lSSL,lSSL,0,bs_dir+'rSSL.dat') #+9*'0'
        # dSSL = ssbits    
        for ib,bit in enumerate(ssbits):
            dSSLs[ori_level,ib] = int(bit) 
            

        for level in range(ori_level,3,-1):

            inds = np.where(dSSLs[level,globz.theshift:-globz.theshift])[0]//2#lowerResolution(np.where(dSSLs[level,globz.theshift:-globz.theshift])[0])
            dSSLs[level-1,globz.theshift+inds] = 1
    
    globz.t1 += time()-start
    for level in range(3,ori_level+1):
        
        
        
        if ENC: #or debug_dec:

            globz.Loc = lrGTs[level]# -mins1+32
    

        lrLoc = lrGTs[level-1]
        LocM = torch_dilate_Loc(lrLoc)#torch.LongTensor(dilate_Loc(lrLoc))#.to(device)
        

        
        # if ENC :#or debug_dec:

        #     globz.Loc = torch.LongTensor(Location).to(device)
            
        maxk1 = 13
            # del Loc_ro
        if not ENC:
            # llbss = np.max(np.where(dSSLs[level,:])[0])+1
            # lbss_in = read_ints(10*np.ones((llbss,),int), bs_dir+'lbss'+str(level)+'.bs')
            
            lbss_in = GR_ENCODE_DECODE(ENC,0,bs_dir+'lbss'+str(level)+'.bs',maxk1=maxk1)
            
            
            with open(bs_dir +'AC'+str(level)+'.b', 'rb') as fin:
                ac_stream_in = fin.read()
        else:
            lbss_in=0
            ac_stream_in=0


        dec_Loc,ac_stream,lbss = get_temps_dests2(ENC,nn_model = nn_model,bs_dir=bs_dir,level=level,ori_level=ori_level,dSSLs=dSSLs,LocM=LocM,lbss_in=lbss_in,ac_stream_in=ac_stream_in,device=device)
        
        if ENC:
            # write_ints(lbss, 10*np.ones((len(lbss),),int), bs_dir+'lbss'+str(level)+'.bs')
    
            GR_ENCODE_DECODE(ENC,lbss,bs_dir+'lbss'+str(level)+'.bs',maxk1=maxk1)
    
    
            with open(bs_dir +'AC'+str(level)+'.b', 'wb') as fout:
                fout.write(ac_stream)
        if not ENC:
            lrGTs[level] = dec_Loc#+mins1#-32
    
    

    
    end = time()
    time_spent = end - start
    nmins = int(time_spent//60)
    nsecs = int(np.round(time_spent-nmins*60))
    print('time spent: ' + str(nmins) + 'm ' + str(nsecs) + 's')
    
    
    if not ENC:
        dec_GT = lrGTs[level].cpu().numpy()
    else:
        dec_GT = 0
    
    return dec_GT,time_spent
        
    

      