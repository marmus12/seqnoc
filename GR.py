#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:42:52 2022

@author: root
"""
from usefuls import write_bits,dec2bin3,read_bits,bin2dec2,get_dir_size
import numpy as np
import os

def GR_ENCODE_DECODE(ENC,ints,bs_path,maxk1=10):
    
    
    nkbits = int(np.ceil(np.log2(maxk1)))
    if ENC:
        maxi = np.max(ints)
        nints = len(ints)
        k1s = range(maxk1+1)
        lbss = np.zeros((maxk1+1,),int)

        for k1 in k1s:
            lbss[k1]=k1*nints+np.sum(ints//(2**k1)) + nints    
        
        k = np.argmin(lbss)   
        bs = dec2bin3(k,nkbits)
        kx2 = 2**k
        for i in ints:
            q = i//kx2
            bs += q*'1'+'0' + dec2bin3(i%kx2,k)


        
        write_bits(bs+'1', bs_path)
    else:
        
        bs = read_bits(bs_path)[0:-1]
        k = bin2dec2(bs[0:nkbits])
        maxi = 2**maxk1
        kx2 = 2**k
        nrem = len(bs[nkbits:])
        ib = nkbits
        ii=0
        dints = np.zeros((1000000,),int)
        while(nrem>0):           
            for ib1 in range(ib,ib+maxi):
                if bs[ib1]=='0':
                    q = ib1-ib
                    ib = ib1+1
                    break
            r = bin2dec2(bs[ib:(ib+k)])
            dints[ii]=q*kx2+r
            ii+=1
            ib+=k
            nrem = nrem - (q+1+k)
        return dints[0:ii]
            
            
def block_GR_ED(ENC,ints,bs_path,maxk1=10,block_size=50):
    
    nbnb = 8  # ! number of blocks cannot exceed 2**nbnb
    nkbits = int(np.ceil(np.log2(maxk1)))
    if not ENC:
        maxi = 2**maxk1        
        ib=0

    if ENC:

        nints = len(ints)
        nblocks = np.ceil(nints/block_size).astype(int)

        bs = dec2bin3(nblocks, nbnb)
        k1s = range(maxk1+1)
        # bks = np.zeros((nblocks,),np.int8)
        lbss = np.zeros((maxk1+1,),int)
        # bks=[]
        for i_block in range(nblocks):
            bints = ints[i_block*block_size:(i_block+1)*block_size]
            nbints = len(bints)


            for k1 in k1s:
                lbss[k1]=k1*nbints+np.sum(bints//(2**k1)) + nbints    
        
            bk = np.argmin(lbss) 
            # bks.append(bk)
            bs+=dec2bin3(bk,nkbits)
            
        # bs = dec2bin3(k,nkbits)
            kx2 = 2**bk
            for i in bints:
                q = i//kx2
                bs += q*'1'+'0' + dec2bin3(i%kx2,bk)

        write_bits(bs+'1', bs_path)
        # return bks
    else:


        bs = read_bits(bs_path)[0:-1]
        lenbs = len(bs)
        nblocks = bin2dec2(bs[0:nbnb])
        ib+=nbnb
        ii=0
        dints = np.zeros((500000,),int)

        # for i_block in range(nblocks):
        while(ib<lenbs):

            bk = bin2dec2(bs[ib:(nkbits+ib)])
            ib+=nkbits

            kx2 = 2**bk

            while(ib<lenbs):           
                for ib1 in range(ib,ib+maxi):
                    if bs[ib1]=='0':
                        q = ib1-ib
                        ib = ib1+1
                        break
                r = bin2dec2(bs[ib:(ib+bk)])
                dints[ii]=q*kx2+r
                ii+=1
                # ib+=bk+
                ib+=bk
                if ii%block_size==0:
                    break

        return dints[0:ii]        
        
        
        
        
        
        
        
if __name__=="__main__":
    
    
    
    nints = 40
    bs_path = 'bs.bs'
    ints = np.random.random_integers(0,127,nints)
    maxk1 = np.ceil(np.log2(np.max(ints))).astype(int)
    block_GR_ED(1,ints,bs_path,maxk1)
    dec_ints = block_GR_ED(0,0,bs_path,maxk1)  
    
    # print(ints)
    # print(dec_ints)
    assert(np.prod(ints==dec_ints))
    
    CL = os.path.getsize(bs_path)*8
    bpi = CL/nints
    print('bpi: '+str(bpi))
    # else:
        
        
        
        
    #     return dints