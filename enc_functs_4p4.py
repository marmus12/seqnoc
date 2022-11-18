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

from usefuls import in1d_index,plt_imshow,write_ints,read_ints,write_bits,read_bits,dec2bin4,dec2bin2_fast,bin2dec2,ints2bs,bs2ints

# import tensorflow.compat.v1 as tf1
import globz
from time import time

# from dec2bin import dec2bin
from runlength import RLED
from GR import GR_ENCODE_DECODE
#from GR import block_GR_ED

     

def get_temps_dests2(ENC=True,nn_model ='dec',bs_dir=None,level=0,ori_level=0,LocM=0,lbss_in=0,ac_stream_in=0,device=0):

    ksize = nn_model.ksize
    delta = (ksize-2)//2

    cpud = torch.device('cpu')
    iBBr = 0
    theshift = globz.theshift

    LocM = LocM+theshift



    maxX,maxY,maxZ = torch.max(LocM,axis=0).values+theshift#+1
        
    VolM = torch.zeros((maxY,maxX,maxZ)).to(device)
    # try:
    VolM[LocM[:,1],LocM[:,0],LocM[:,2]]=2
    




    # %% Find sections in Loc
    # lSSL = maxY.item()#+1
 
    ac_stream = bytes()
    if ENC:
        
        
        globz.Loc = globz.Loc+theshift#torch.LongTensor([theshift ,theshift ,theshift ]).to(device)
        
        VolL = torch.zeros((maxY,maxX,maxZ)).to(device)
        VolL[globz.Loc[:,1],globz.Loc[:,0],globz.Loc[:,2]]=1


        # SSL2 = np.zeros((lSSL,),'int8')
        # SSL2[globz.Loc[:,1].cpu()]=1

        
        # if level==ori_level :
        #     # st5=time()
        #     ssbits = ''
        #     for ssbit in SSL2:
        #         ssbits=ssbits+str(ssbit)
            # globz.t5+=time()-st5
            

            
            # RLED(ssbits,lSSL,lSSL,1,bs_dir+'rSSL.dat')
            
    # else:    
        

        
        # SSL2 = dSSLs[level,:]
        
        
        


    # %% Find sections in LocM


    if not ENC:        
        lLocM = LocM.shape[0] 
        globz.Loc = torch.zeros((lLocM ,3),dtype=torch.int16).to(device)


    dssx,dssz = torch.div(maxX-delta,2,rounding_mode='trunc'),torch.div(maxZ-delta,2,rounding_mode='trunc')
    masks=[]
    masks.append(torch.zeros((dssx-delta//2,dssz-delta//2)).bool().to(device))
    masks.append(torch.zeros((dssx-delta//2,dssz-delta//2)).bool().to(device))
    masks.append(torch.zeros((dssx-delta//2,dssz-delta//2)).bool().to(device))
    masks.append(torch.zeros((dssx-delta//2,dssz-delta//2)).bool().to(device))
    # dsBWTrueM = torch.zeros((dssx,dssz)).bool().to(device)
    VolM4 = torch.zeros((1,4,maxX,maxZ)).to(device)

    # shiftsz = (0,1,0,1) 
    b2dmult = torch.Tensor([[[[8,4],[2,1]]]]).to(device)
    lbss = np.zeros((4*(maxY+1),),'int16')
    total_lbs = 0
    Wp2c = (1-torch.tril(torch.ones((16,17))))#.to(device)
    # Wp2c[:,-1]=0
    if ENC:
        mx4 = torch.div(maxX,4,rounding_mode='trunc')
        mz4 = torch.div(maxZ,4,rounding_mode='trunc')
        szx = 4*mx4
        szz = 4*mz4
    
        Tmasks = []
        Tmasks.append(torch.zeros([maxX,maxZ]).bool().to(device))
        patch0 = torch.zeros([4,4]).bool()
        patch0[0:2,0:2] = True
        
        Tmasks[0][delta:(delta+szx),delta:(delta+szz)] = patch0.repeat((mx4,mz4))[0:(maxX-delta),0:(maxZ-delta)]
        # inds0x,inds0z = torch.where(Tmask0)
        ######################################
        Tmasks.append(torch.zeros([maxX,maxZ]).bool().to(device))
        patch1 = torch.zeros([4,4]).bool()
        patch1[0:2,2:] = True
        
        Tmasks[1][delta:(delta+szx),delta:(delta+szz)] = patch1.repeat((mx4,mz4))[0:(maxX-delta),0:(maxZ-delta)]  
        # inds1x,inds1z = torch.where(Tmask1)    
    ################################################3    
        Tmasks.append(torch.zeros([maxX,maxZ]).bool().to(device))
        patch2 = torch.zeros([4,4]).bool()
        patch2[2:,0:2] = True
        
        Tmasks[2][delta:(delta+szx),delta:(delta+szz)] = patch2.repeat((mx4,mz4))[0:(maxX-delta),0:(maxZ-delta)]      
    
    
    ilb=0
    for icPC in range(theshift,maxY):
        
        if icPC%300==0:
            print('icPC:' + str(icPC))


            
        # if SSL2[icPC]: #ncand>0
            
     
        if ENC:
            
            VolM4[0,0:2,:,:] = VolL[(icPC-2):(icPC),:,:] 
            BWT0 = VolL[icPC:(icPC+1),:,:]
            BWT = BWT0[:,delta:-delta,delta:-delta].unsqueeze(0)
            BWT0 = BWT0.squeeze()                


        VolM4[0,2:4,:,:] = VolM[icPC:(icPC+2),:,:]
        


        dsBWTrueM  = (VolM[icPC,delta:(2*dssx):2,delta:(2*dssz):2]/2).bool()
        



        masks[0][0::2,0::2] = dsBWTrueM[0::2,0::2]
        masks[1][0::2,1::2] = dsBWTrueM[0::2,1::2]
        masks[2][1::2,0::2] = dsBWTrueM[1::2,0::2]                
        masks[3][1::2,1::2]  = dsBWTrueM[1::2,1::2]              
        
        nzmasks = []
        nzmasks.append(torch.nonzero(masks[0]))
        nzmasks.append(torch.nonzero(masks[1]))
        nzmasks.append(torch.nonzero(masks[2]))
        nzmasks.append(torch.nonzero(masks[3]))
        ncand0 = nzmasks[0].shape[0]
        ncand1 = nzmasks[1].shape[0]
        ncand2 = nzmasks[2].shape[0]
        ncand3 = nzmasks[3].shape[0]
        ncands = [ncand0,ncand1,ncand2,ncand3]
        if ENC:
            symbs_enc = torch.conv2d(BWT,b2dmult,stride=[2,2])#.squeeze(0).squeeze(0)
        
        

        for imask in range(4):
            
            if ncands[imask]>0:
                nn_output = nn_model.forward(VolM4)
                mncand = ncands[imask]
                output_pdf = nn_output.masked_select(masks[imask]).reshape([16,mncand]).transpose(0,1).to(cpud) #ncand0,16
                good_cdf = torch.clamp(torch.matmul(output_pdf,Wp2c),0,1)#.to(cpud)                
                # flat_cdf = good_cdf.flatten()
                flat_pdf = output_pdf.flatten()
                inds0 = torch.LongTensor(range(mncand))*16
                    
                if ENC:
                    tsymbs = symbs_enc.masked_select(masks[imask]).type(torch.int16).to(cpud)     
                    
                    inds = inds0+tsymbs
                    # indsm1 = inds0+tsymbs 
                    # lbs_est = int((-torch.sum(torch.log2(flat_cdf[inds]-flat_cdf[indsm1]))/8).item())  
                    lbs_est = int((-torch.sum(torch.log2(flat_pdf[inds]))/8).item())  
                    
                    
                    byte_stream = torchac.encode_float_cdf(good_cdf, tsymbs, check_input_bounds=True)
                    if imask<3:
                        VolM4[0,2,:,:] =  VolM4[0,2,:,:] + ((BWT0)*2+1-2)*Tmasks[imask]*(VolM4[0,2,:,:]/2) #True zero:1, True one:3
                    


                    
                    
                    lbss[ilb] = len(byte_stream)-lbs_est
                    ilb+=1
                    ac_stream = ac_stream+byte_stream
                else:

                    
                   
                    
                    byte_stream = ac_stream_in[total_lbs:]#(total_lbs+lbs)]
                    dec_syms = torchac.decode_float_cdf(good_cdf,byte_stream)#.to(device)#.tolist()
                    
                    st2  = time()
                    
                    inds = inds0+dec_syms
                    
                    dec_syms = dec_syms.to(device)

                    lbs_est = int((-torch.sum(torch.log2(flat_pdf[inds]))/8).item())  
    
                    lbs = lbs_est+lbss_in[ilb]
                    ilb+=1
                    st3 = time()
                    globz.t2 += st3-st2
                    
                    total_lbs += lbs
                    block_sym = torch.zeros((mncand,4),dtype=torch.bool).to(device)
                    block_sym[:,0] = torch.div(dec_syms,8,rounding_mode='trunc')
                    block_sym[:,1] =  torch.div((dec_syms%8),4,rounding_mode='trunc')
                    block_sym[:,2] = torch.div((dec_syms%4),2,rounding_mode='trunc')
                    block_sym[:,3] = dec_syms%2
                    
                    st4 = time()
                    globz.t3 += st4-st3
                    # nzmask = 
                    ixzs = nzmasks[imask]*2+delta#[:,0]
                    # izs = (nzmask*2+2)#[:,1]
                # shxs = torch.cat((ixs,ixs,ixs+1,ixs+1)) #4xnsyms
                # shzs = torch.cat((izs,izs+1,izs,izs+1)) #4xnsyms
                    shxzs1 = ixzs[block_sym[:,0],:]      
                    # shzs1 = izs[block_sym[:,0,0]]  
                                                               #0,1
                    shxzs2 = ixzs[block_sym[:,1],:]      #      1,1       
                    # shzs2 = izs[block_sym[:,0,1]]  
                    
                    shxzs3 = ixzs[block_sym[:,2],:]      
                    # shzs3 = izs[block_sym[:,1,0]]  
                    
                    shxzs4 = ixzs[block_sym[:,3],:]      
                    # shzs4 = izs[block_sym[:,1,1]]  
                    st5 = time()
                    globz.t4 += st5-st4
                    
                    shxs = torch.cat((shxzs1 ,shxzs2 ,shxzs3+1,shxzs4+1))[:,0] #4xnsyms
                    shzs = torch.cat((shxzs1 ,shxzs2+1,shxzs3,shxzs4+1))[:,1] #4xnsyms
                    pcandxs = torch.cat((ixzs ,ixzs ,ixzs+1,ixzs+1))[:,0]  #4xnsyms
                    pcandzs = torch.cat((ixzs ,ixzs+1,ixzs,ixzs+1))[:,1]  #4xnsyms      
                    st6 = time()
                    globz.t5 += st6-st5
                    
                    # if imask<3:                        
                    VolM4[0,2,pcandxs,pcandzs] = 1 # true zeros
                    VolM4[0,2,shxs,shzs] = 3 #true ones
                    npts = shxs.shape[0]
                    
                    st7 = time()
                    globz.t6 += st7-st6
                # for ip in range(npts):
                # new_pts = icPC*torch.ones((npts,3),dtype=torch.int).to(device)
                    globz.Loc[iBBr:(iBBr+npts),0] = shxs
                    globz.Loc[iBBr:(iBBr+npts),1] = icPC
                    globz.Loc[iBBr:(iBBr+npts),2] = shzs
                    # globz.Loc[iBBr:(iBBr+npts),:] = torch.cat((shxs,icPC*torch.ones((npts,)),shzs))
                    iBBr +=npts


            
        if not ENC:
            VolM4[0,0,:] = VolM4[0,1,:]
            VolM4[0,1,:] = VolM4[0,2,:]==3
            
            


    if ENC:    
        dec_Loc = 0
    if not(ENC) :#and not(for_train):
        dec_Loc =  (globz.Loc[0:iBBr,:]-theshift)#.cpu().numpy()


    return dec_Loc,ac_stream,lbss[0:ilb]
    
        


    

    
def ENCODE_DECODE(ENC,bs_dir,nn_model,ori_level=0,GT=0,device = torch.device('cuda')):
    

    # cpud = torch.device('cpu')
    
    # tGT = torch.Tensor(GT).to(device)
    
    start = time()      

    # globz.theshift=6
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
        # dSSLs = 0
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
            
        # mins11 = lrmms[ori_level,1]
        # maxes11 = lrmms[ori_level,4]        
        # lSSL = maxes11+2*globz.theshift#+1#+32+10#+40
                
       ##get dssls     
        # dSSLs  = np.zeros((ori_level+1,4500),int)

        # ssbits =''# 32*'0'
        # ssbits = RLED('',lSSL,lSSL,0,bs_dir+'rSSL.dat') #+9*'0'
        # dSSL = ssbits    
        # for ib,bit in enumerate(ssbits):
        #     dSSLs[ori_level,ib] = int(bit) 
            

        # for level in range(ori_level,3,-1):

        #     inds = np.where(dSSLs[level,globz.theshift:-globz.theshift])[0]//2#lowerResolution(np.where(dSSLs[level,globz.theshift:-globz.theshift])[0])
        #     dSSLs[level-1,globz.theshift+inds] = 1
    
    globz.t1 += time()-start
    for level in range(3,ori_level+1):
        
        
        
        if ENC: #or debug_dec:

            globz.Loc = lrGTs[level]# -mins1+32
    

        lrLoc = lrGTs[level-1]
        LocM = torch_dilate_Loc(lrLoc)#torch.LongTensor(dilate_Loc(lrLoc))#.to(device)
        


        maxk1 = 5
            # del Loc_ro
        if not ENC:
            # llbss = np.max(np.where(dSSLs[level,:])[0])+1
            # lbss_in = read_ints(10*np.ones((llbss,),int), bs_dir+'lbss'+str(level)+'.bs')
            
            lbss_in = GR_ENCODE_DECODE(ENC,0,bs_dir+'lbss'+str(level)+'.bs',maxk1=maxk1)-30
            # lbss_in = block_GR_ED(ENC, 0,bs_dir+'lbss'+str(level)+'.bs',maxk1=maxk1,block_size=gr_block_size)
            
            with open(bs_dir +'AC'+str(level)+'.b', 'rb') as fin:
                ac_stream_in = fin.read()
        else:
            lbss_in=0
            ac_stream_in=0


        dec_Loc,ac_stream,lbss = get_temps_dests2(ENC,nn_model = nn_model,bs_dir=bs_dir,level=level,ori_level=ori_level,LocM=LocM,lbss_in=lbss_in,ac_stream_in=ac_stream_in,device=device)
        
        if ENC:
            # write_ints(lbss, 10*np.ones((len(lbss),),int), bs_dir+'lbss'+str(level)+'.bs')
    
            GR_ENCODE_DECODE(ENC,lbss+30,bs_dir+'lbss'+str(level)+'.bs',maxk1=maxk1)
            # block_GR_ED(ENC, lbss,bs_dir+'lbss'+str(level)+'.bs',maxk1=maxk1,block_size=gr_block_size)
    
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
        
    

      
