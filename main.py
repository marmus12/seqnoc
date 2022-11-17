# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from pcloud_functs import pcshow_pyto, pcread, slice_to_8
from pcloud_functs import lowerResolution as lowRes
from torch_train_utils import writeCNNtobs
import numpy as np
import os, inspect
from shutil import copyfile,rmtree
import time
import torch
# import random
from datetime import datetime

from glob import glob
# from scipy.io import loadmat
from usefuls import compare_Locations,plt_imshow,get_dir_size,show_time_spent

import globz
globz.init()
#from collections import OrderedDict

from torch_train_utils import torch_criterion,compute_model_CL,collect_opt_data7,collect_opt_data8,collect_opt_data9,torch_criterion_mm
import argparse



parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_root', default='seqnoc_outputs/',
                    help='where to save the results, bitstreams',required=False)
parser.add_argument('--ply_dir', default='/media/emre/Data1/DATA/loot/loot/Ply/',
                    help='directory containing the input sequence',required=False)
parser.add_argument('--Nfr', type=int, default=100,
                    help='number of frames you wanna encode',required=False)
parser.add_argument('--bs', type=int, default=10000,
                    help='training batch size',required=False)
parser.add_argument('--noptfrs', type=int, default=5,
                    help='number of training frames',required=False)
parser.add_argument('--fourphase', type=bool, default=True,
                    help='four phases or single phase',required=False)
parser.add_argument('--cnn', type=str, default='default',
                    help='CNN structure. should be one of: default, SNHM,SNLM, SNLM2',required=False)
parser.add_argument('--decode', type=bool, default=True,
                    help='Whether you wanna check the decoding or not',required=False)

args = parser.parse_args()



output_root = args.output_root
ply_dir = args.ply_dir
batch_size  =args.bs
noptframes = args.noptfrs
fourpass = args.fourphase
print(args)
#%%##
if not( os.path.exists(output_root)):
    os.makedirs(output_root)
slice_pc = 0 # for high resolution single frames (sequences not supported)

filepaths = glob(ply_dir+'*.ply')[1:(args.Nfr+1)]
device = torch.device('cuda')


the_criterion = torch_criterion_mm


perm='automatic'
if perm!='automatic':
    perm=[2,0,1] # sweeping dim is in the middle
    

init_lr = 0.001
min_lr =  0.0003 
lr_drop_rate=0.5
patience = 20 #

if args.cnn == 'default':
    from pytorch_models import SeqNOCdefault as model_type 
elif args.cnn == 'SNLM':
    from pytorch_models import SNLM as model_type 
elif args.cnn == 'SNLM2':
    from pytorch_models import SNLM2 as model_type 
elif args.cnn == 'SNHM':
    from pytorch_models import SNHM as model_type 



if fourpass:
    from enc_functs_4p3 import ENCODE_DECODE
else:
    from enc_functs_pt6 import ENCODE_DECODE
decode=args.decode


#%%##################################

     
mdl = model_type(device)
ctx_type=mdl.ctx_type
CL_mdl,nws=compute_model_CL(mdl.state_dict(), 32)


print('num weights : '+str(nws))
print('CL_mdl: '+str(CL_mdl))

curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")

output_dir = output_root + curr_date + '/'

os.mkdir(output_dir)
curr_file = inspect.getfile(inspect.currentframe()) 
copyfile(curr_file,output_dir + curr_date + "__" + curr_file.split("/")[-1]) 
    
#%%#####


if perm=='automatic':
    shortest_dim = np.argmin(np.max(pcread(filepaths[0]),0))
    other_dims = np.setdiff1d([0,1,2],[shortest_dim])
    perm = [other_dims[0],shortest_dim,other_dims[1]]
    print('based on dimensions, permutation is chosen as:'+str(perm))
    
nframes = len(filepaths)

opt_frids = list(np.arange(0,nframes,nframes//noptframes)[0:noptframes])
print(opt_frids)



#%%

print("Collect ctxs and counts...")
start = time.time()


if slice_pc:
    GT = pcread(filepaths[0])
    GT = GT[:,perm]
    sGTs = slice_to_8(GT)
    ctxs,counts,nu_ctxs = collect_opt_data9(sGTs,[perm],device,ctx_type=ctx_type,fourpass=fourpass)
else:
    ctxs,counts,nu_ctxs = collect_opt_data7(filepaths,opt_frids,[perm],device,ctx_type=ctx_type,fourpass=fourpass)

print('number of unique contexts:'+str(nu_ctxs))
collect_time = time.time()-start
show_time_spent(collect_time)
# 
#%%
print("Optimize...")
start = time.time()

optimizer = torch.optim.Adam(params=mdl.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=lr_drop_rate,patience=patience)


n_eiters = int(np.ceil(nu_ctxs/batch_size))
print('number of iters in one epoch:'+str(n_eiters))

epoch = 0
while(True):
    for itr in range(n_eiters):
        
        bctxs = ctxs[batch_size*itr:batch_size*(itr+1)]
        bcounts = counts[batch_size*itr:batch_size*(itr+1)]
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = mdl(bctxs)
        loss = the_criterion(outputs, bcounts)
        loss.backward()
        optimizer.step()
        # titr+=1
    scheduler.step(loss)
    # print statistics
    learning_rate = optimizer.param_groups[0]['lr']
    if epoch%3==0:
        print('epoch: ' + str(epoch)+'  loss: '+ str(np.around(loss.item(),4)))
        print('learning_rate: '+str(learning_rate))

    if learning_rate<min_lr:
        break
    epoch+=1

        
train_time = time.time()-start
show_time_spent(train_time)
#%%



#%% 
del ctxs,counts
torch.cuda.empty_cache() 
#%% 

print("ENCODE DECODE")


root_bs_dir = output_dir + 'bss/'
     

bpvs = np.zeros((nframes))
mean_bpv = 0; mean_mdl_bpv=0; mean_tse=0; mean_tsd=0
dec_errors = np.zeros((nframes),dtype='bool')
times = np.zeros((nframes,2))
num_fails=0
ifile_start=0
#%%

GT = pcread(filepaths[0])

ori_level = int(np.ceil(np.log2(np.max(GT))))

if slice_pc: # single pc only
    bs_dir = root_bs_dir
    GT = pcread(filepaths[0])
    npts = GT.shape[0]
    GT = GT[:,perm]
    sGTs = slice_to_8(GT)
     
    time_spente = 0
    time_spentd = 0
    CL=0
    for i_slice,sGT in enumerate(sGTs):
        if len(sGT)>0:
            print('doing slice: ' + str(i_slice) )
            os.mkdir(bs_dir)  
            _,te = ENCODE_DECODE(1,bs_dir,mdl,ori_level-1,sGT)
            time_spente+=te
            CL+= get_dir_size(bs_dir)

            if decode:
                dec_sGT,td = ENCODE_DECODE(0,bs_dir,mdl,ori_level-1)
                time_spentd+=td
                TP,FP,FN=compare_Locations(dec_sGT,sGT)
                if (FP.shape[0]+FN.shape[0]) != 0:
                    # dec_errors[ifile] = 1
                    num_fails+=1
                print('num_fails:'+str(num_fails))            
        
            rmtree(bs_dir)            
            
    mdl_bpv = CL_mdl/npts  
    mean_mdl_bpv =mdl_bpv
    bpv = (CL+CL_mdl)/npts
    bpvs[0] = bpv
    print('bpv: '+str(np.around(bpv,5)))
    print('mdl_bpv: '+str(np.around(mdl_bpv,9)))
    print('filepath:'+filepaths[0])
    print('input level:'+str(ori_level))
    print('enc_dec_times:'+str(np.around(time_spente,4))+'  '+str(np.around(time_spentd,4)))
    times[0,0] = time_spente
    times[0,1] = time_spentd
    ifile=0
    
else:

    np.random.shuffle(filepaths)
    os.mkdir(root_bs_dir) 
    writeCNNtobs(mdl.state_dict(),root_bs_dir+'CNN.bs')
    for ifile,filepath in enumerate(filepaths[ifile_start:]):
        print('##################################################')
        ifile=ifile+ifile_start
        print('ifile: '+ str(ifile))
        ori_GT = pcread(filepath)

        npts = ori_GT.shape[0]
        
        #%###################################    
        # for iperm in range(nperms):
        bs_dir = root_bs_dir+'bs'+str(ifile)+'/'
        os.mkdir(bs_dir)   
        
        GT = ori_GT[:,perm]    
        
        
        _,time_spente = ENCODE_DECODE(1,bs_dir,mdl,ori_level,GT)
        
        
        CL = get_dir_size(bs_dir)
        mdl_bpv = CL_mdl/(nframes*npts)
        bpv = CL/npts+mdl_bpv#(CL+CL_mdl/nframes)/npts
        
        bpvs[ifile] = bpv#.append(bpv)
        
        print('bpv: '+str(np.around(bpv,5)))
        print('mdl_bpv: '+str(np.around(mdl_bpv,9)))
        print('filepath:'+filepath)
        print('input level:'+str(ori_level))
        mean_mdl_bpv = (mean_mdl_bpv*ifile+mdl_bpv)/(ifile+1)
        mean_bpv = (mean_bpv*ifile+bpv)/(ifile+1)
        print('mean_bpv:'+str(np.around(mean_bpv,4)))
        
        if decode:
            dec_GT,time_spentd = ENCODE_DECODE(0,bs_dir,mdl,ori_level)
            
            TP,FP,FN=compare_Locations(dec_GT,GT)
            if (FP.shape[0]+FN.shape[0]) != 0:
                dec_errors[ifile] = 1
                num_fails+=1
            print('num_fails:'+str(num_fails))
            print('bpv: '+str(np.around(bpv,5)))
            
            print('filepath:'+filepath)
            print('input level:'+str(ori_level))
            print('dec:')
            show_time_spent(time_spentd)
            times[ifile,:] = time_spente,time_spentd
            mean_tse = (mean_tse*ifile+time_spente)/(ifile+1)
            mean_tsd = (mean_tsd*ifile+time_spentd)/(ifile+1)
            
            print('mean_times:'+str(np.around(mean_tse+(collect_time+train_time)/nframes,4))+'  '+str(np.around(mean_tsd,4)))

        

#%%#############################
enctime0 = (collect_time)/nframes
enctime1 = (train_time)/nframes

while (bpvs[ifile]==0):
    ifile=ifile-1
    
lines = []
lines.append('num files encoded:'+str(ifile+1))
if not(slice_pc):
    lines.append('ply_dir: ' +ply_dir)
lines.append('num_fails: '+str(num_fails))
lines.append('perm:'+str(perm))
lines.append(curr_date)
lines.append('mean_bpv:'+str(np.around(np.mean(bpvs[0:(ifile+1)],0),4)))
lines.append('mean_mdl_bpv:'+str(np.around(mean_mdl_bpv,4)))

meantimes = np.mean(times[0:(ifile+1),:],0)

lines.append('enc time: ' + str(np.around(enctime0,1))+ '+' + str(np.around(enctime1,1))+'+'+str(np.around(meantimes[0],1))+'s')
lines.append('total enc time: ' + str(np.around(enctime0+enctime1+meantimes[0],1)))
if decode:
    lines.append('dec time: '+ str(np.around(meantimes[1],1))+'s')
# lines.append('mean_ enc time (collect opt included):'+str(np.around(meantimes[0]+enctime0,2)))
lines.append('ctx count collection time:')
lines.append(show_time_spent(collect_time))
lines.append('Optimization time:')
lines.append(show_time_spent(train_time))
lines.append('num_epochs: '+str(epoch))
lines.append('end loss:'+str(np.around(loss.item(),4)))

lines.append('batch_size: '+str(batch_size))
# lines.append('npbits : '+str(npbits))
lines.append('num weights : '+str(nws))
lines.append('CL_mdl: '+str(CL_mdl))
lines.append('number of unique contexts:'+str(nu_ctxs))
lines.append('model: ' +model_type.__name__)
if ifile==0:
    lines.append(filepaths[0])
with open(output_dir+'info.txt', 'w') as f:
    for line in lines:
        print(line)
        f.write(line)
        f.write('\n')

with open(output_dir+'fpaths.txt', 'w') as f:
    for line in filepaths:
        f.write(line)
        f.write('\n')

np.save(output_dir+'bpvs.npy', bpvs[0:(ifile+1)])
np.save(output_dir+'filepaths.npy', filepaths[0:(ifile+1)])



