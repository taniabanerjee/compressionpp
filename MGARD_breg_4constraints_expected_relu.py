#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math

import os

#from PIL import Image
from tqdm import tqdm

import adios2 as ad2
import xgc4py
import nanopq
#from torchsummary import summary


# In[2]:


def draw(frame):
    x = np.linspace(0, 38, 39)
    y = np.linspace(0, 38, 39)
    X, Y = np.meshgrid(x, y)
    plt.imshow(frame, origin='lower')
    #plt.imshow(frame, origin='upper')
    plt.colorbar()
    plt.contour(X, Y, frame, 5, origin='image', colors='white', alpha=0.5)


# In[3]:


eb = ["1e12","5e12","1e13","5e13","7e13","8e13","1e14","2e14","3e14","4e14","5e14","6e14","7e14","8e14","9e14","1e15", "2e15","3e15","4e15","5e15","6e15", "7e15","8e15","9e15","1e16","2e16","3e16","4e16","5e16","6e16","7e16","8e16","9e16","1e17"]


# In[4]:


## timestep 420 for compression ratio reference
timestep = 420
compression_ratio_mgard_uniform = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/mgard/v2_{}/uniform/compression_ratio_mgard_uniform.npy'.format(timestep))
print(compression_ratio_mgard_uniform[:21])
compression_ratio_mgard_uniform_reduced = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))
print(compression_ratio_mgard_uniform_reduced)


# In[4]:


# MGARD

timestep = 1000

f0f_rmse_mgard_uniform = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/mgard/v2_{}/uniform/f0f_rmse_mgard.npy'.format(timestep))
f0f_rel_rmse_ornl_mgard_uniform = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/mgard/v2_{}/uniform/f0f_rel_rmse_ornl_mgard.npy'.format(timestep))
f0f_rel_rmse_mine_mgard_uniform = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/mgard/v2_{}/uniform/f0f_rel_rmse_mine_mgard.npy'.format(timestep))
QoI_rmse_mgard_uniform = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/mgard/v2_{}/uniform/QoI_rmse_mgard.npy'.format(timestep))
QoI_rel_rmse_ornl_mgard_uniform = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/mgard/v2_{}/uniform/QoI_rel_rmse_ornl_mgard.npy'.format(timestep))
QoI_rel_rmse_mine_mgard_uniform = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/mgard/v2_{}/uniform/QoI_rel_rmse_mine_mgard.npy'.format(timestep))
compression_ratio_mgard_uniform = np.load('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/results/mgard/v2_{}/uniform/compression_ratio_mgard_uniform.npy'.format(timestep))
print(compression_ratio_mgard_uniform.shape)
print(compression_ratio_mgard_uniform[:26])


# In[7]:


'''
#timestep 420
save_idx = np.array([0, 6, 8, 11, 15, 16, 17, 18 ,19, 20])
print(save_idx.shape)
'''
#timestep 100
#save_idx = np.array([0, 7, 14, 17, 18 ,20, 21, 22, 23, 24])
save_idx = np.array([18])
print(save_idx.shape)


# In[8]:


reduced_eb = []
f0f_rel_rmse_ornl_mgard_uniform_reduced = np.zeros((save_idx.shape[0]))
QoI_rel_rmse_ornl_mgard_uniform_reduced = np.zeros((save_idx.shape[0],6))
compression_ratio_mgard_uniform_reduced = np.zeros((save_idx.shape[0]))

for i in range(len(save_idx)):
    idx = save_idx[i]
    print('eb, compression ratio: ', eb[idx], compression_ratio_mgard_uniform[idx])
    reduced_eb.append(eb[idx])
    f0f_rel_rmse_ornl_mgard_uniform_reduced[i] = f0f_rel_rmse_ornl_mgard_uniform[idx]
    QoI_rel_rmse_ornl_mgard_uniform_reduced[i] = QoI_rel_rmse_ornl_mgard_uniform[idx]
    compression_ratio_mgard_uniform_reduced[i] = compression_ratio_mgard_uniform[idx]

reduced_eb = np.array(reduced_eb)
reduced_eb.shape
print(reduced_eb)


# In[9]:


import os
import errno

dirname = '/gpfs/alpine/csc143/proj-shared/tania/XGC_2/results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/'.format(timestep)
os.makedirs(os.path.dirname(dirname), exist_ok=True)


# In[10]:


timestep


# In[11]:


np.save('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/reduced_eb.npy'.format(timestep), reduced_eb)
np.save('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/f0f_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep), f0f_rel_rmse_ornl_mgard_uniform_reduced)
np.save('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/QoI_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep), QoI_rel_rmse_ornl_mgard_uniform_reduced)
np.save('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep), compression_ratio_mgard_uniform_reduced)


# In[12]:


reduced_eb = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/reduced_eb.npy'.format(timestep))
f0f_rel_rmse_ornl_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/f0f_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
QoI_rel_rmse_ornl_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/QoI_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
compression_ratio_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))
reduced_eb.shape, f0f_rel_rmse_ornl_mgard_uniform_reduced.shape, QoI_rel_rmse_ornl_mgard_uniform_reduced.shape


# In[13]:


reduced_eb


# In[15]:


# New results

# PD

'''
fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

axis.plot(compression_ratio_mgard_uniform_reduced, f0f_rel_rmse_ornl_mgard_uniform_reduced, '-^r', label='MGARD')
axis.plot(compression_ratio_mgard_uniform[:26], f0f_rel_rmse_ornl_mgard_uniform[:26], '-*', label='MGARD (uniform)')

axis.set_xlabel('Compression ratio', fontsize=18, fontweight='bold')
axis.set_ylabel('NRMSE', fontsize=18, fontweight='bold')
#axis.spines['top'].set_visible(False)
#axis.spines['right'].set_visible(False)

axis.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3), fontsize=14)
#legend_properties = {'size':'16','weight':'bold'}
#axis.legend(loc="upper left", bbox_to_anchor=(0.12, 1.18), ncol = 2, frameon=False, prop=legend_properties)
#axis.legend(loc="upper left", fontsize=12)
plt.xticks(fontsize=13, fontweight ='bold')
plt.yticks(fontsize=13, fontweight ='bold')
'''


# In[16]:


# New results

# Avg QoI

'''
fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

axis.plot(compression_ratio_mgard_uniform_reduced, np.average(QoI_rel_rmse_ornl_mgard_uniform_reduced, axis=(1)), '-^r', label='MGARD')
axis.plot(compression_ratio_mgard_uniform[:26], np.average(QoI_rel_rmse_ornl_mgard_uniform[:26], axis=(1)), '-*', label='MGARD (uniform)')

axis.set_xlabel('Compression ratio', fontsize=18, fontweight='bold')
axis.set_ylabel('NRMSE', fontsize=18, fontweight='bold')
#axis.spines['top'].set_visible(False)
#axis.spines['right'].set_visible(False)

axis.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3), fontsize=14)
#legend_properties = {'size':'16','weight':'bold'}
#axis.legend(loc="upper left", bbox_to_anchor=(0.12, 1.18), ncol = 2, frameon=False, prop=legend_properties)
#axis.legend(loc="upper left", fontsize=12)
plt.xticks(fontsize=13, fontweight ='bold')
plt.yticks(fontsize=13, fontweight ='bold')
'''


# In[17]:


reduced_eb = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/reduced_eb.npy'.format(timestep))
f0f_rel_rmse_ornl_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/f0f_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
QoI_rel_rmse_ornl_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/QoI_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
compression_ratio_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))
reduced_eb.shape, f0f_rel_rmse_ornl_mgard_uniform_reduced.shape, QoI_rel_rmse_ornl_mgard_uniform_reduced.shape


# In[18]:


timestep


# In[19]:


s_val = 0

for i in range(len(reduced_eb)):
    filename = '/gpfs/alpine/csc143/proj-shared/ljm/MGARD_2/MGARD-SMC/build/v2_{0}/uniform/d3d_coarse_v2_{0}.bp.mgard.4d.s{1}.{2}'.format(timestep,s_val,reduced_eb[i])
    with ad2.open(filename, 'r') as f:
        f0_g = f.read('i_f_4d')
    print(f0_g.shape)
    print(type(f0_g[0][0][0][0]))
    np.save('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/MGARD_uniform_{}.npy'.format(timestep, reduced_eb[i]), f0_g)
    del f0_g


# In[3]:


with ad2.open('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/dataset/xgc.mesh.bp', 'r') as f:
    nnodes = int(f.read('n_n', ))
    ncells = int(f.read('n_t', ))
    rz = f.read('rz')
    conn = f.read('nd_connect_list')
    psi = f.read('psi')
    nextnode = f.read('nextnode')
    epsilon = f.read('epsilon')
    node_vol = f.read('node_vol')
    node_vol_nearest = f.read('node_vol_nearest')
    psi_surf = f.read('psi_surf')
    surf_idx = f.read('surf_idx')
    surf_len = f.read('surf_len')

r = rz[:,0]
z = rz[:,1]
print (nnodes)


# In[4]:


timestep = 1000
with ad2.open('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/dataset/d3d_coarse_v2_{}.bp'.format(timestep), 'r') as f:
    i_f = f.read('i_f')
i_f.shape


# In[5]:


xgcexp = xgc4py.XGC('/gpfs/alpine/csc143/proj-shared/ljm/XGC_2/dataset')


# In[6]:


f0_f = np.copy(i_f[0])
ndata = f0_f.shape[0]
print(ndata)
f0_inode1 = 0

#vol, vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge, f0_grid_vol, mu_vp_vol= xgcexp.f0_diag_vol(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_f)
vol, vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge, f0_grid_vol, mu_vp_vol= xgcexp.f0_diag_vol_origdim(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_f)

np.save('./vol.npy', vol.astype(float))
np.save('./vth.npy', vth.astype(float))
np.save('./vp.npy', vp.astype(float))
np.save('./mu_qoi.npy', mu_qoi.astype(float))
np.save('./vth2.npy', vth2.astype(float))
np.save('./f0_grid_vol.npy', f0_grid_vol.astype(float))
np.save('./mu_vp_vol.npy', mu_vp_vol.astype(float))

# In[7]:


del f0_f


# In[8]:


def qoi_numerator_para(f0_f, vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge):
    den = f0_f.astype(np.float128) * vol
    s_den = np.sum(den, axis=(1,2))
    V2 = vol * vth[:,np.newaxis,np.newaxis] * vp[np.newaxis,np.newaxis,:]
    #upar = f0_f * V2
    upar = f0_f.astype(np.float128) * vol * vth[:,np.newaxis,np.newaxis] * vp[np.newaxis,np.newaxis,:]
    s_u_par = np.sum(upar, axis=(1,2))
    upara = s_u_par/s_den
    tper = f0_f.astype(np.float128) * vol * 0.5 * mu_qoi[np.newaxis,:,np.newaxis] * vth2[:,np.newaxis,np.newaxis] * ptl_mass
    s_t_perp = np.sum(tper, axis=(1,2))
    tperp = s_t_perp/s_den/sml_e_charge
    
    upar_ = upara/vth
    en   = 0.5 * (vp[np.newaxis,:].astype(np.float128) - upar_[:,np.newaxis])**2
    T_par_ = f0_f.astype(np.float128) * vol * en[:,np.newaxis,:] * vth2[:,np.newaxis,np.newaxis] * ptl_mass
    tpara = 2.0*np.sum(T_par_, axis=(1,2), dtype=np.float128)/s_den/sml_e_charge
    
    return upara, upar_, en


# In[9]:


def qoi_numerator(f0_f, vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge):
    den = f0_f.astype(np.float128) * vol
    s_den = np.sum(den, axis=(1,2))
    V2 = vol * vth[:,np.newaxis,np.newaxis] * vp[np.newaxis,np.newaxis,:]
    #upar = f0_f * V2
    upar = f0_f.astype(np.float128) * vol * vth[:,np.newaxis,np.newaxis] * vp[np.newaxis,np.newaxis,:]
    s_u_par = np.sum(upar, axis=(1,2))
    upara = s_u_par/s_den
    tper = f0_f.astype(np.float128) * vol * 0.5 * mu_qoi[np.newaxis,:,np.newaxis] * vth2[:,np.newaxis,np.newaxis] * ptl_mass
    s_t_perp = np.sum(tper, axis=(1,2))
    tperp = s_t_perp/s_den/sml_e_charge
    
    upar_ = upara/vth
    en   = 0.5 * (vp[np.newaxis,:].astype(np.float128) - upar_[:,np.newaxis])**2
    T_par_ = f0_f.astype(np.float128) * vol * en[:,np.newaxis,:] * vth2[:,np.newaxis,np.newaxis] * ptl_mass
    tpara = 2.0*np.sum(T_par_, axis=(1,2), dtype=np.float128)/s_den/sml_e_charge
    
    return s_den, upara, tperp, tpara


# In[12]:


n_phi = 8
ndata = 16395
f0_inode1 = 0

den_f = np.zeros((n_phi, ndata))
upara_f = np.zeros((n_phi, ndata))
tperp_f = np.zeros((n_phi, ndata))
tpara_f = np.zeros((n_phi, ndata))

for iphi in range(n_phi):
    den_f[iphi], upara_f[iphi], tperp_f[iphi], tpara_f[iphi], _, _ = xgcexp.f0_diag(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=np.copy(i_f[iphi]))

# In[17]:


n_phi = 8
ndata = 16395
f0_inode1 = 0

den_m = np.zeros((n_phi, ndata))
upara_m = np.zeros((n_phi, ndata))
tperp_m = np.zeros((n_phi, ndata))
tpara_m = np.zeros((n_phi, ndata))

for iphi in range(n_phi):
    den_m[iphi], upara_m[iphi], tperp_m[iphi], tpara_m[iphi] = qoi_numerator(i_f[iphi], vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge)


# In[14]:


print(np.array_equal(den_f, den_m))
print(np.array_equal(upara_f, upara_m))
print(np.array_equal(tperp_f, tperp_m))
print(np.array_equal(tpara_f, tpara_m))


# In[10]:


def qoi_numerator_matrices(f0_f, vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge):
    den = f0_f.astype(np.float128) * vol
    s_den = np.sum(den, axis=(1,2))
    V2 = vol * vth[:,np.newaxis,np.newaxis] * vp[np.newaxis,np.newaxis,:]
    #upar = f0_f * V2
    upar = f0_f.astype(np.float128) * vol * vth[:,np.newaxis,np.newaxis] * vp[np.newaxis,np.newaxis,:]
    s_u_par = np.sum(upar, axis=(1,2))
    upara = s_u_par/s_den
    tper = f0_f.astype(np.float128) * vol * 0.5 * mu_qoi[np.newaxis,:,np.newaxis] * vth2[:,np.newaxis,np.newaxis] * ptl_mass
    V3 = vol * 0.5 * mu_qoi[np.newaxis,:,np.newaxis] * vth2[:,np.newaxis,np.newaxis] * ptl_mass
    s_t_perp = np.sum(tper, axis=(1,2))
    tperp = s_t_perp/s_den/sml_e_charge
    
    upar_ = upara/vth
    en   = 0.5 * (vp[np.newaxis,:].astype(np.float128) - upar_[:,np.newaxis])**2
    T_par_ = f0_f.astype(np.float128) * vol * en[:,np.newaxis,:] * vth2[:,np.newaxis,np.newaxis] * ptl_mass
    #V4 = 2.0*(vol * en[:,np.newaxis,:] * vth2[:,np.newaxis,np.newaxis] * ptl_mass)
    V4 = vol * ((vp[np.newaxis,np.newaxis,:]**2)) * vth2[:,np.newaxis,np.newaxis] * ptl_mass
    tpara = 2.0*np.sum(T_par_, axis=(1,2), dtype=np.float128)/s_den/sml_e_charge
    
    return V2, V3, V4


# In[11]:


class Nonnegativity(nn.Module):
    def __init__(self):
        super(Nonnegativity, self).__init__()
        
        self.layer1 = nn.ReLU()
        
    def forward(self, x):
        
        x = self.layer1(x) + 100.0

        return x


# In[12]:


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# In[13]:


def get_nonnegative(model, test_loader, num_instance, feature_dim):
    #reconstructions = np.zeros((78*16,num_coeffs))
    outputs_cpu = []
        
    for batch in test_loader:
        data = batch[0]
        data = data.to(device)
        outputs = model(data)
        outputs_cpu.append(outputs.cpu().data.numpy())
        #reconstructions[count] = outputs.cpu().numpy()
        #count = count + 1
        
    #print('network model output type: ', outputs.dtype)
    
    output_vectors = np.zeros((num_instance,feature_dim), dtype=np.float64)
    count = 0
    for i in range(len(outputs_cpu)):
        for j in range(len(outputs_cpu[i])):
            output_vectors[count] = np.float64(outputs_cpu[i][j])
            count = count + 1  
    
    return output_vectors


# In[14]:


timestep


# In[15]:


reduced_eb = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/reduced_eb.npy'.format(timestep))
reduced_eb.shape,reduced_eb


# In[21]:


''' manually assign positivity like relu

# remove negative value with relu in recon
tailname = 'MGARD_uniform'
print(tailname)

for eb in reduced_eb:    
    recon = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/{}_{}.npy'.format(timestep, tailname, eb))
    print('min recon {} and shape {}'.format(np.min(recon), recon.shape))
    for p in range(8):
        for idx in range(16395):
            for r in range(39):
                for c in range(39):
                    if recon[p][idx][r][c] < 0:
                        recon[p][idx][r][c] = 100.0
                    else:
                        recon[p][idx][r][c] = recon[p][idx][r][c] + 100.0
                        
    print('eb {}, min {}'.format(eb, np.min(recon)))
    np.save('./results/MGARD_Lagrange_expected/v2_{}/{}_{}_nonngegative_relu.npy'.format(timestep, tailname, eb), recon)
'''


# In[33]:


# remove negative value in recon

Batch_Size = 256

tailname = 'MGARD_uniform'
print(tailname)

torch.set_default_dtype(torch.float64)

for eb in reduced_eb:    
    recon = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/{}_{}.npy'.format(timestep, tailname, eb))
    print('eb {} before relu min recon {}'.format(eb, np.min(recon)))
    print('eb {} recon shape'.format(eb), recon.shape)
    
    for p in range(8): # 8 planes
    
        recon_p_tensor = torch.Tensor(recon[p].reshape(16395,1521))
        if p == 0:
            print('tensor type: ', recon_p_tensor.dtype)
        
        training_data = torch.utils.data.TensorDataset(recon_p_tensor)
        #del recon_p_tensor
        training_loader = DataLoader(training_data,
                                     batch_size=Batch_Size,
                                     shuffle=False,
                                     pin_memory=True)
        
        model = Nonnegativity()
        device = get_device()
        model.to(device)
        #print(model)        
        
        recon_p_nonnegatvie = get_nonnegative(model, training_loader, 16395, 1521)
        recon_p_nonnegatvie = recon_p_nonnegatvie.reshape(16395,39,39)
        #print(type(recon_l_nonnegatvie[0][0][0]))
        recon[p] = recon_p_nonnegatvie        
        
    print('eb {}, after relu min recon {}'.format(eb, np.min(recon)))    
    np.save('./results/MGARD_Lagrange_expected/v2_{}/{}_{}_nonngegative_relu.npy'.format(timestep, tailname, eb), recon)


# In[34]:


import os
import errno

dirname = './results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/'.format(timestep)
os.makedirs(os.path.dirname(dirname), exist_ok=True)


# In[16]:


def rmse_error(x, y):

    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)

    return np.sqrt(mse)


# In[17]:


reduced_eb = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/reduced_eb.npy'.format(timestep))
timestep, reduced_eb.shape,reduced_eb


# In[18]:


# minimize negative of dual with 4 constraints
# 분모에 D를 전부 우변으로 옮김

from numpy.linalg import inv
from numpy.linalg import pinv

#latent_dims = [5, 10, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 300]

#step_size = 0.01

n_phi = 8
ndata = 16395
f0_inode1 = 0

den_f = np.zeros((n_phi, ndata))
upara_f = np.zeros((n_phi, ndata))
tperp_f = np.zeros((n_phi, ndata))
tpara_f = np.zeros((n_phi, ndata))

tailname = 'MGARD_uniform'
print('timestpe: ', timestep)
print('tailname: ', tailname)

print('Start')
print('Compute original QoIs')
for iphi in range(n_phi):
    den_f[iphi], upara_f[iphi], tperp_f[iphi], tpara_f[iphi] = qoi_numerator(i_f[iphi], vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge)

np.save('./den_f.npy', den_f)
np.save('./upara_f.npy', upara_f)
np.save('./tperp_f.npy', tperp_f)
np.save('./tpara_f.npy', tpara_f)

#for eb_idx in range(len(reduced_eb)):
eb_idx = 0
eb = reduced_eb[eb_idx]
recon = np.load('./results/MGARD_Lagrange_expected/v2_{}/{}_{}_nonngegative_relu.npy'.format(timestep, tailname, eb))
print('eb {} recon shape'.format(eb), recon.shape)
print('min recon: ', np.min(recon))

recon_breg = np.zeros((8,16395,39,39))
lagranges = np.zeros((8,16395,4))

Tp = np.zeros((8,16395))
for p in range(8):
    Tp[p] = (sml_e_charge * tpara_f[p] + (vth2 * ptl_mass * ((upara_f[p]/vth)**2)))
np.save('./Tpara.npy', Tp.astype(float))
for p in range(8):
    V2, V3, V4 = qoi_numerator_matrices(i_f[p], vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge)
    np.save('./V2.npy', V2.astype(float))
    np.save('./V3.npy', V3.astype(float))
    np.save('./V4.npy', V4.astype(float))
    print('V2 V3 V4 shape: ', V2.shape, V3.shape, V4.shape)
    
    D = den_f[p]
    U = upara_f[p]
    Tperp = tperp_f[p]
    #Tpara = tpara_f[p]
    Tpara = (sml_e_charge * tpara_f[p] + (vth2 * ptl_mass * ((upara_f[p]/vth)**2)))
    np.save('./V2.npy', V2.astype(float))
    
    print('density shape: ', D.shape)       
    
    
    #L2_den_latent = []
    #L2_upara_latent = []
    #L2_tperp_latent = []
    
    L2_den_all = []
    L2_upara_all = []
    L2_tperp_all = []
    L2_tpara_all = []
    
    count_unLag = 0
    node_unconv = []
    
    for idx in range(16395):
        recon_one = recon[p][idx]
        gradients = np.zeros((4), dtype = np.float128)
        hessians = np.zeros((4,4), dtype = np.float128)
        
        #initialize lambda
        lambdas = np.zeros((4), dtype = np.float128)
        
        L2_den = []
        L2_upara = []
        L2_tperp = []
        L2_tpara = []
        
        #K = lambdas[0]*vol + lambdas[1]*V2/s_den_d +lambdas[2]*V3/(s_den_d*sml_e_charge)
        #print(K.shape)
        
        count = 0
        aD = D[idx]*sml_e_charge
        #V2_D = V2[idx]/D[idx]
        #V3_D = V3[idx]/(D[idx]*sml_e_charge)
        #V4_D = V4[idx]/(D[idx]*sml_e_charge)
        
        while(1):
            #print('count: ', count)
            
            K = np.zeros((39,39))
            
            K = lambdas[0]*vol[idx] + lambdas[1]*V2[idx] + lambdas[2]*V3[idx] + lambdas[3]*V4[idx]
            
            
            if count > 0:
                breg_result = recon_one*np.exp(-K)
                
                update_D = np.sum(breg_result*vol[idx], dtype=np.float128)
                update_U = np.sum(breg_result*V2[idx], dtype=np.float128)/D[idx]
                update_Tperp = np.sum(breg_result*V3[idx], dtype=np.float128)/(aD)
                update_Tpara = np.sum(breg_result*V4[idx], dtype=np.float128)/D[idx]
                
                L2_D = (update_D - D[idx])**2
                L2_U = (update_U - U[idx])**2
                L2_Tperp = (update_Tperp - Tperp[idx])**2
                L2_Tpara = (update_Tpara - Tpara[idx])**2
                
                L2_den.append(L2_D)
                L2_upara.append(L2_U)
                L2_tperp.append(L2_Tperp)
                L2_tpara.append(L2_Tpara)
                
                
                distortion_rate = np.max(recon_one)/np.max(breg_result)
                
                if ((count == 20) and ((L2_den[-1] - L2_den[0] > 0) or (L2_upara[-1] - L2_upara[0] > 0)                        or (L2_tperp[-1] - L2_tperp[0] > 0) or (L2_tpara[-1] - L2_tpara[0] > 0))):
                    print('node {} is not converged'.format(idx))
                    count_unLag = count_unLag + 1
                    node_unconv.append(idx)
                    #L2_den_all.append(np.array(L2_den))
                    #L2_upara_all.append(np.array(L2_upara))
                    #L2_tperp_all.append(np.array(L2_tperp))
                    #L2_tpara_all.append(np.array(L2_tpara))
                
                if (count == 20):
                    recon_breg[p][idx] = breg_result
                    lagranges[p][idx] = lambdas
                    #print('node {} count: '.format(idx), count)
                    L2_den_all.append(np.array(L2_den))
                    L2_upara_all.append(np.array(L2_upara))
                    L2_tperp_all.append(np.array(L2_tperp))
                    L2_tpara_all.append(np.array(L2_tpara))
                    
                    if idx % 2000 == 0:
                        print('node {} finished'.format(idx))                           
                
                    break
                                      
                                      
            gradients[0] = -np.sum((recon_one.astype(np.float128)*vol[idx]*np.exp(-K)), dtype=np.float128) + D[idx]
            gradients[1] = -np.sum((recon_one.astype(np.float128)*V2[idx]*np.exp(-K)), dtype=np.float128) + U[idx]*D[idx]                
            gradients[2] = -np.sum((recon_one.astype(np.float128)*V3[idx]*np.exp(-K)), dtype=np.float128) + Tperp[idx]*aD
            gradients[3] = -np.sum((recon_one.astype(np.float128)*V4[idx]*np.exp(-K)), dtype=np.float128) + Tpara[idx]*D[idx]
            
            hessians[0][0] = np.sum(recon_one.astype(np.float128)*(np.power(vol[idx],2))*np.exp(-K), dtype=np.float128)
            hessians[0][1] = np.sum(recon_one.astype(np.float128)*vol[idx]*V2[idx]*np.exp(-K), dtype=np.float128)                
            hessians[0][2] = np.sum(recon_one.astype(np.float128)*vol[idx]*V3[idx]*np.exp(-K), dtype=np.float128)
            hessians[0][3] = np.sum(recon_one.astype(np.float128)*vol[idx]*V4[idx]*np.exp(-K), dtype=np.float128)
            
            hessians[1][0] = hessians[0][1]
            hessians[1][1] = np.sum(recon_one.astype(np.float128)*(np.power(V2[idx],2))*np.exp(-K), dtype=np.float128)                
            hessians[1][2] = np.sum(recon_one.astype(np.float128)*V2[idx]*V3[idx]*np.exp(-K), dtype=np.float128)
            hessians[1][3] = np.sum(recon_one.astype(np.float128)*V2[idx]*V4[idx]*np.exp(-K), dtype=np.float128)
            
            hessians[2][0] = hessians[0][2]
            hessians[2][1] = hessians[1][2]
            hessians[2][2] = np.sum(recon_one.astype(np.float128)*(np.power(V3[idx],2))*np.exp(-K), dtype=np.float128)
            hessians[2][3] = np.sum(recon_one.astype(np.float128)*V3[idx]*V4[idx]*np.exp(-K), dtype=np.float128)
            
            hessians[3][0] = hessians[0][3]
            hessians[3][1] = hessians[1][3]
            hessians[3][2] = hessians[2][3]
            hessians[3][3] = np.sum(recon_one.astype(np.float128)*(np.power(V4[idx],2))*np.exp(-K), dtype=np.float128)
            
            hessians = np.float64(hessians)
            
            #lambdas = lambdas - inv(hessians) @ gradients
            
            try:
                lambdas = lambdas - inv(hessians) @ gradients
            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    print('Singular occurs at eb{} p{} idx{}'.format(eb, p, idx))
                    lambdas = lambdas - pinv(hessians) @ gradients
                else:
                    raise
            
            
            lambdas = lambdas.astype(np.float128)
            
            count = count + 1
        
    print('max original f: ', np.max(i_f[p]))
    print('eb {} max plane {} recon breg: '.format(eb, p), np.max(recon_breg[p]))
    recon_rmse = rmse_error(i_f[p], recon[p])
    breg_rmse = rmse_error(i_f[p], recon_breg[p])
    print('recon PD rmse: {:.2e}, breg PD rmse: {:.2e}'.format(recon_rmse, breg_rmse))
    print('Number of nodes that are not converged', count_unLag)
    #L2_den_all = np.array(L2_den_all)
    #L2_upara_all = np.array(L2_upara_all)
    #L2_tperp_all = np.array(L2_tperp_all)
    
np.save('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/{}_{}_breg_denorm.npy'.format(timestep, tailname, eb), recon_breg)
np.save('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/{}_{}_lagranges.npy'.format(timestep, tailname, eb), lagranges)


# In[57]:


plt.plot(L2_tperp_all[11250])


# In[58]:


plt.plot(L2_tpara_all[11250])


# In[26]:


lagranges.shape, type(lagranges[0][0][0])


# In[19]:


# MGARD uniform

reduced_eb = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/reduced_eb.npy'.format(timestep))
compression_ratio_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))
print(reduced_eb.shape)


timestep, reduced_eb.shape, compression_ratio_mgard_uniform_reduced.shape, compression_ratio_mgard_uniform_reduced


# In[20]:


# Compression ratio with fixed cost with lagrange orig 64bits
# Measure 1 plane ratio
# sizes are in bits

data_size = 16395.*39.*39.*8.*8.
lagrange_cost = 16395.*4.*8.*8.
print('data size: ', data_size)

compression_ratio_fixed_cost_MGARD_breg = np.zeros((len(compression_ratio_mgard_uniform_reduced)))

for i in range(len(compression_ratio_mgard_uniform_reduced)):
    compression_ratio_fixed_cost_MGARD_breg[i] = data_size/((data_size/compression_ratio_mgard_uniform_reduced[i]) + lagrange_cost)

print('compression_ratio_fixed_cost_MGARD_breg: ', compression_ratio_fixed_cost_MGARD_breg)

tailname = 'MGARD_uniform_breg_relu'
np.save('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname), compression_ratio_fixed_cost_MGARD_breg)


# In[21]:


min_lag = np.zeros((len(compression_ratio_mgard_uniform_reduced)))
max_lag = np.zeros((len(compression_ratio_mgard_uniform_reduced)))

tailname = 'MGARD_uniform'

for eb_idx in range(len(reduced_eb)):
    eb = reduced_eb[eb_idx]
    lagranges = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/{}_{}_lagranges.npy'.format(timestep, tailname, eb))
    min_lag[eb_idx] = np.min(lagranges)
    max_lag[eb_idx] = np.max(lagranges)
    print('min max of lagranges {}: '.format(eb), np.min(lagranges), np.max(lagranges))
    
print('total min max: ', np.min(min_lag), np.max(max_lag))


# In[21]:


# Store lagrange with PQ

# minimize negative of dual with 4 constraints
# 분모에 D를 전부 우변으로 옮김

from numpy.linalg import inv
from numpy.linalg import pinv

#step_size = 0.01

n_phi = 8
ndata = 16395
f0_inode1 = 0

pq_var = [64,256,1024]
pq_bits = [6,8,10]

pq_dic = []

tailname = 'MGARD_uniform'

print('Start')

for b in range(len(pq_bits)):
    print('PQ bits: ', pq_bits[b])
    
    for eb_idx in range(len(reduced_eb)):
        eb = reduced_eb[eb_idx]
        print('eb: ', eb)
        recon = np.load('./results/MGARD_Lagrange_expected/v2_{}/{}_{}_nonngegative_relu.npy'.format(timestep, tailname, eb))
        breg_orig = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/{}_{}_breg_denorm.npy'.format(timestep, tailname, eb))
        print('eb {} recon shape'.format(eb), recon.shape)
        print('min recon: ', np.min(recon))
        
        recon_breg = np.zeros((recon.shape[0],16395,39,39))
        print('recon breg eb {} shape'.format(eb), recon_breg.shape)
        lagranges = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/{}_{}_lagranges.npy'.format(timestep, tailname, eb))
        lagranges = lagranges.astype(np.float32)
        print('Lagranges shape of eb {}'.format(eb), lagranges.shape)
        print('Type of lagrange parameters: ', type(lagranges[0][0][0]))
        
        for p in range(8):
            V2, V3, V4 = qoi_numerator_matrices(i_f[p], vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge)
            print('V2 V3 V4 shape: ', V2.shape, V3.shape, V4.shape)
            
            def train_quantizer():
                seed = np.random.randint(100)                
                pq = nanopq.PQ(M=4, Ks=pq_var[b], verbose=True)
                pq.fit(vecs=lagranges[p], iter=20, seed=seed)
                return pq
            
            pq = train_quantizer()
            
            if ((p == 0) and (eb_idx == 0)):
                pq_dic.append(pq.codewords)
                
            lagranges_quan = pq.encode(vecs=lagranges[p])
            lagranges_dequan = pq.decode(codes=lagranges_quan)
            if ((p == 0) and (eb_idx == 0)):
                print('lagranges_dequan shape: ', lagranges_dequan.shape)
            
            for idx in range(16395):
                recon_one = recon[p][idx].astype(np.float128)
                lambdas = lagranges_dequan[idx].astype(np.float128)
                
                K = np.zeros((39,39))     
                K = lambdas[0]*vol[idx] + lambdas[1]*V2[idx] + lambdas[2]*V3[idx] + lambdas[3]*V4[idx]
                breg_result = recon_one*np.exp(-K)
                recon_breg[p][idx] = breg_result.astype(np.float64)
                
            print('max original f: ', np.max(i_f[p]))
            print('eb {} max plane {} recon breg: '.format(eb, p), np.max(recon_breg[p]))
            #recon_rmse = rmse_error(i_f[p], recon[l])
            breg_orig_rmse = rmse_error(i_f[p], breg_orig[p])
            breg_rmse = rmse_error(i_f[p], recon_breg[p])
            orig_rmse = rmse_error(i_f[p], recon[p])
            #print('recon PD rmse: {:.2e}, breg PD rmse: {:.2e}'.format(recon_rmse, breg_rmse))
            print('Orig PD rmse: {:.2e}, breg PQ PD rmse: {:.2e}'.format(orig_rmse, breg_rmse))
            print('breg orig PD rmse: {:.2e}, breg PQ PD rmse: {:.2e}'.format(breg_orig_rmse, breg_rmse))
            
            
            
        np.save('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/{}_{}_breg_Lagpara_pq_{}bits_denorm.npy'.format(timestep, tailname, eb, pq_bits[b]), recon_breg)


# In[63]:


pq_dic[0].shape


# In[22]:


len(compression_ratio_mgard_uniform_reduced)


# In[23]:


# Compression ratio with fixed cost with lagrange orig 64bits
# Measure 1 plane ratio
# sizes are in bits

data_size = 16395.*39.*39.*8.*8.
lagrange_cost = 16395.*4.*8.*8.
print('data size: ', data_size)

compression_ratio_fixed_cost_MGARD_breg = np.zeros((len(compression_ratio_mgard_uniform_reduced)))

for i in range(len(compression_ratio_mgard_uniform_reduced)):
    compression_ratio_fixed_cost_MGARD_breg[i] = data_size/((data_size/compression_ratio_mgard_uniform_reduced[i]) + lagrange_cost)

print('compression_ratio_fixed_cost_MGARD_breg: ', compression_ratio_fixed_cost_MGARD_breg)

tailname = 'MGARD_uniform_breg_relu'
np.save('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname), compression_ratio_fixed_cost_MGARD_breg)


# In[6]:


timestep = 700
timestep


# In[7]:


reduced_eb = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/reduced_eb.npy'.format(timestep))
f0f_rel_rmse_ornl_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/f0f_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
QoI_rel_rmse_ornl_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/QoI_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
compression_ratio_mgard_uniform_reduced = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))
reduced_eb.shape, f0f_rel_rmse_ornl_mgard_uniform_reduced.shape, QoI_rel_rmse_ornl_mgard_uniform_reduced.shape


# In[8]:


# Compression ratio with fixed cost with lagrange orig 64bits
# Measure 1 plane ratio
# sizes are in bits

data_size = 16395.*39.*39.*8.*8.
print('data size: ', data_size)

pq_var = [64,256,1024]
pq_bits = [6,8,10]

compression_ratio_fixed_cost_MGARD_breg = np.zeros((len(pq_bits), len(compression_ratio_mgard_uniform_reduced)))

for b in range(len(pq_bits)):
    lagrange_cost = 16395.*4.*pq_bits[b]
    pq_dics_qoi = 4.*pq_var[b]*4.*8.
    
    for i in range(len(compression_ratio_mgard_uniform_reduced)):
        compression_ratio_fixed_cost_MGARD_breg[b][i] = data_size/((data_size/compression_ratio_mgard_uniform_reduced[i]) + lagrange_cost + pq_dics_qoi)
        
    print('compression_ratio_fixed_cost_MGARD_breg_Lagpara_{}bits: '.format(pq_bits[b]), compression_ratio_fixed_cost_MGARD_breg[b])
    
    
    tailname = 'MGARD_uniform_breg_relu_Lagpara_pq_{}bits'.format(pq_bits[b])
    np.save('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname), compression_ratio_fixed_cost_MGARD_breg[b])


# In[ ]:





# In[22]:


# MGARD

timestep = 420

s_val = -1
f0f_rmse_mgard_s_minus1 = np.load('./results/mgard/v2_{}/s{}/f0f_rmse_mgard.npy'.format(timestep,s_val))
f0f_rel_rmse_ornl_mgard_s_minus1 = np.load('./results/mgard/v2_{}/s{}/f0f_rel_rmse_ornl_mgard.npy'.format(timestep,s_val))
f0f_rel_rmse_mine_mgard_s_minus1 = np.load('./results/mgard/v2_{}/s{}/f0f_rel_rmse_mine_mgard.npy'.format(timestep,s_val))
QoI_rmse_mgard_s_minus1 = np.load('./results/mgard/v2_{}/s{}/QoI_rmse_mgard.npy'.format(timestep,s_val))
QoI_rel_rmse_ornl_mgard_s_minus1 = np.load('./results/mgard/v2_{}/s{}/QoI_rel_rmse_ornl_mgard.npy'.format(timestep,s_val))
QoI_rel_rmse_mine_mgard_s_minus1 = np.load('./results/mgard/v2_{}/s{}/QoI_rel_rmse_mine_mgard.npy'.format(timestep,s_val))
compression_ratio_mgard_s_minus1 = np.load('./results/mgard/v2_{}/s{}/compression_ratio_mgard.npy'.format(timestep,s_val))

f0f_rmse_mgard_s_minus1_nodedist = np.load('./results/mgard/v2_{}/s{}/f0f_rmse_mgard_nodedist.npy'.format(timestep,s_val))
f0f_rel_rmse_ornl_mgard_s_minus1_nodedist = np.load('./results/mgard/v2_{}/s{}/f0f_rel_rmse_ornl_mgard_nodedist.npy'.format(timestep,s_val))
f0f_rel_rmse_mine_mgard_s_minus1_nodedist = np.load('./results/mgard/v2_{}/s{}/f0f_rel_rmse_mine_mgard_nodedist.npy'.format(timestep,s_val))
QoI_rmse_mgard_s_minus1_nodedist = np.load('./results/mgard/v2_{}/s{}/QoI_rmse_mgard_nodedist.npy'.format(timestep,s_val))

del s_val
s_val = 0
f0f_rmse_mgard_s_0 = np.load('./results/mgard/v2_{}/s{}/f0f_rmse_mgard2.npy'.format(timestep,s_val))
f0f_rel_rmse_ornl_mgard_s_0 = np.load('./results/mgard/v2_{}/s{}/f0f_rel_rmse_ornl_mgard2.npy'.format(timestep,s_val))
f0f_rel_rmse_mine_mgard_s_0 = np.load('./results/mgard/v2_{}/s{}/f0f_rel_rmse_mine_mgard2.npy'.format(timestep,s_val))
QoI_rmse_mgard_s_0 = np.load('./results/mgard/v2_{}/s{}/QoI_rmse_mgard2.npy'.format(timestep,s_val))
QoI_rel_rmse_ornl_mgard_s_0 = np.load('./results/mgard/v2_{}/s{}/QoI_rel_rmse_ornl_mgard2.npy'.format(timestep,s_val))
QoI_rel_rmse_mine_mgard_s_0 = np.load('./results/mgard/v2_{}/s{}/QoI_rel_rmse_mine_mgard2.npy'.format(timestep,s_val))
compression_ratio_mgard_s_0 = np.load('./results/mgard/v2_{}/s{}/compression_ratio_mgard2.npy'.format(timestep,s_val))

f0f_rmse_mgard_s_0_nodedist = np.load('./results/mgard/v2_{}/s{}/f0f_rmse_mgard_nodedist2.npy'.format(timestep,s_val))
f0f_rel_rmse_ornl_mgard_s_0_nodedist = np.load('./results/mgard/v2_{}/s{}/f0f_rel_rmse_ornl_mgard_nodedist2.npy'.format(timestep,s_val))
f0f_rel_rmse_mine_mgard_s_0_nodedist = np.load('./results/mgard/v2_{}/s{}/f0f_rel_rmse_mine_mgard_nodedist2.npy'.format(timestep,s_val))

f0f_rmse_mgard_uniform = np.load('./results/mgard/v2_{}/uniform/f0f_rmse_mgard.npy'.format(timestep))
f0f_rel_rmse_ornl_mgard_uniform = np.load('./results/mgard/v2_{}/uniform/f0f_rel_rmse_ornl_mgard.npy'.format(timestep))
f0f_rel_rmse_mine_mgard_uniform = np.load('./results/mgard/v2_{}/uniform/f0f_rel_rmse_mine_mgard.npy'.format(timestep))
QoI_rmse_mgard_uniform = np.load('./results/mgard/v2_{}/uniform/QoI_rmse_mgard.npy'.format(timestep,s_val))
QoI_rel_rmse_ornl_mgard_uniform = np.load('./results/mgard/v2_{}/uniform/QoI_rel_rmse_ornl_mgard.npy'.format(timestep))
QoI_rel_rmse_mine_mgard_uniform = np.load('./results/mgard/v2_{}/uniform/QoI_rel_rmse_mine_mgard.npy'.format(timestep))
compression_ratio_mgard_uniform = np.load('./results/mgard/v2_{}/uniform/compression_ratio_mgard_uniform.npy'.format(timestep))


# In[10]:


timestep = 100
f0f_rel_rmse_ornl_mgard_uniform_reduced_100 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/f0f_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
QoI_rel_rmse_ornl_mgard_uniform_reduced_100 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/QoI_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
compression_ratio_mgard_uniform_reduced_100 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))

timestep = 420
f0f_rel_rmse_ornl_mgard_uniform_reduced_420 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/f0f_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
QoI_rel_rmse_ornl_mgard_uniform_reduced_420 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/QoI_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
compression_ratio_mgard_uniform_reduced_420 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))

timestep = 700
f0f_rel_rmse_ornl_mgard_uniform_reduced_700 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/f0f_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
QoI_rel_rmse_ornl_mgard_uniform_reduced_700 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/QoI_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
compression_ratio_mgard_uniform_reduced_700 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))


timestep = 1000
f0f_rel_rmse_ornl_mgard_uniform_reduced_1000 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/f0f_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
QoI_rel_rmse_ornl_mgard_uniform_reduced_1000 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/QoI_rel_rmse_ornl_mgard_uniform_reduced.npy'.format(timestep))
compression_ratio_mgard_uniform_reduced_1000 = np.load('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/evaluations/compression_ratio_mgard_uniform_reduced.npy'.format(timestep))


# In[33]:


tailname = 'MGARD_uniform_breg_relu'
f0f_rmse_mgard_uniform_breg_64bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rmse_{}.npy'.format(timestep,tailname))
f0f_rel_rmse_ornl_mgard_uniform_breg_64bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
QoI_rmse_mgard_uniform_breg_64bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rmse_{}.npy'.format(timestep,tailname))
QoI_rel_rmse_ornl_mgard_uniform_breg_64bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
compression_ratio_mgard_uniform_breg_64bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname))


# In[35]:


# MGARD pq bits
timestep = 420

pq_bits = 6

tailname = 'MGARD_uniform_breg_relu_Lagpara_pq_{}bits'.format(pq_bits)
f0f_rel_rmse_ornl_mgard_uniform_breg_pq_6bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
QoI_rel_rmse_ornl_mgard_uniform_breg_pq_6bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rel_rmse_ornl_{}.npy'.format(timestep,tailname))

compression_ratio_mgard_uniform_breg_pq_6bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname))

pq_bits = 8

tailname = 'MGARD_uniform_breg_relu_Lagpara_pq_{}bits'.format(pq_bits)
f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rel_rmse_ornl_{}.npy'.format(timestep,tailname))

compression_ratio_mgard_uniform_breg_pq_8bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname))

pq_bits = 10

tailname = 'MGARD_uniform_breg_relu_Lagpara_pq_{}bits'.format(pq_bits)
f0f_rel_rmse_ornl_mgard_uniform_breg_pq_10bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
QoI_rel_rmse_ornl_mgard_uniform_breg_pq_10bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rel_rmse_ornl_{}.npy'.format(timestep,tailname))

compression_ratio_mgard_uniform_breg_pq_10bits = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname))


# In[9]:


# MGARD pq bits

timestep = 100
pq_bits = 8

tailname = 'MGARD_uniform_breg_relu_Lagpara_pq_{}bits'.format(pq_bits)
f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_100 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_100 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rel_rmse_ornl_{}.npy'.format(timestep,tailname))

compression_ratio_mgard_uniform_breg_pq_8bits_100 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname))

timestep = 420
pq_bits = 8

tailname = 'MGARD_uniform_breg_relu_Lagpara_pq_{}bits'.format(pq_bits)
f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_420 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_420 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rel_rmse_ornl_{}.npy'.format(timestep,tailname))

compression_ratio_mgard_uniform_breg_pq_8bits_420 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname))

timestep = 700
pq_bits = 8

tailname = 'MGARD_uniform_breg_relu_Lagpara_pq_{}bits'.format(pq_bits)
f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_700 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_700 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rel_rmse_ornl_{}.npy'.format(timestep,tailname))

compression_ratio_mgard_uniform_breg_pq_8bits_700 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname))

timestep = 1000
pq_bits = 8

tailname = 'MGARD_uniform_breg_relu_Lagpara_pq_{}bits'.format(pq_bits)
f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_1000 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/f0f_rel_rmse_ornl_{}.npy'.format(timestep,tailname))
QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_1000 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/QoI_rel_rmse_ornl_{}.npy'.format(timestep,tailname))

compression_ratio_mgard_uniform_breg_pq_8bits_1000 = np.load('./results/MGARD_Lagrange_expected/v2_{}/den_upara_tperp_tpara/evaluations/compression_ratio_fixed_cost_{}.npy'.format(timestep,tailname))


# In[21]:


# New results

# PD

fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

#axis.plot(compression_ratio_mgard_uniform[:20], f0f_rel_rmse_ornl_mgard_uniform[:20], '-^k', label='MGARD (uniform)')
#axis.plot(compression_ratio_mgard_s_minus1[:16], f0f_rel_rmse_ornl_mgard_s_minus1[:16], '-^m', label='MGARD (s=-1)')
#axis.plot(compression_ratio_mgard_uniform_breg_64bits, f0f_rel_rmse_ornl_mgard_uniform_breg_64bits, '-^', label='MGARD (uniform) + postprocessing (no PQ)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_6bits, f0f_rel_rmse_ornl_mgard_uniform_breg_pq_6bits, '-^', label='MGARD (uniform) + postprocessing (PQ 6bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits, f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits, '-^', label='MGARD (uniform) + postprocessing (PQ 8bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_10bits, f0f_rel_rmse_ornl_mgard_uniform_breg_pq_10bits, '-^', label='MGARD (uniform) + postprocessing (PQ 10bits)')
#axis.plot(compression_ratio_mgard_s_minus1[:16], f0f_rel_rmse_ornl_mgard_s_minus1[:16], '-^m', label='MGARD (s=-1)')

axis.plot(compression_ratio_mgard_uniform_reduced_100, f0f_rel_rmse_ornl_mgard_uniform_reduced_100, '-^', label='Timestpe: 100, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_420, f0f_rel_rmse_ornl_mgard_uniform_reduced_420, '-^', label='Timestpe: 420, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_700, f0f_rel_rmse_ornl_mgard_uniform_reduced_700, '-^', label='Timestpe: 700, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_1000, f0f_rel_rmse_ornl_mgard_uniform_reduced_1000, '-^', label='Timestpe: 1000, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_100, f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_100, '-^', label='Timestpe: 100, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_420, f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_420, '-^', label='Timestpe: 420, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_700, f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_700, '-^', label='Timestpe: 700, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_1000, f0f_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_1000, '-^', label='Timestpe: 1000, MGARD (uniform) + postprocessing (PQ 8bits)')


axis.set_xlabel('Compression ratio', fontsize=14)
axis.set_ylabel('NRMSE', fontsize=15)

axis.legend(loc="lower right", bbox_to_anchor=(2.35, 0.2), fontsize=14)
#plt.subplots_adjust(wspace=0.25)
plt.suptitle('PD',fontsize=23)


# In[19]:


# New results

# PD

fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

#axis.plot(compression_ratio_mgard_uniform[:20], np.average(QoI_rel_rmse_ornl_mgard_uniform[:20], axis=(1)), '-^k', label='MGARD (uniform)')
#axis.plot(compression_ratio_mgard_s_minus1[:16], np.average(QoI_rel_rmse_ornl_mgard_s_minus1[:16], axis=(1)), '-^m', label='MGARD (s=-1)')
#axis.plot(compression_ratio_mgard_uniform_breg_64bits, np.average(QoI_rel_rmse_ornl_mgard_uniform_breg_64bits, axis=(1)), '-^', label='MGARD (uniform) + postprocessing (no PQ)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_6bits, np.average(QoI_rel_rmse_ornl_mgard_uniform_breg_pq_6bits, axis=(1)), '-^', label='MGARD (uniform) + postprocessing (PQ 6bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits, np.average(QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits, axis=(1)), '-^', label='MGARD (uniform) + postprocessing (PQ 8bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_10bits, np.average(QoI_rel_rmse_ornl_mgard_uniform_breg_pq_10bits, axis=(1)), '-^', label='MGARD (uniform) + postprocessing (PQ 10bits)')
axis.plot(compression_ratio_mgard_uniform_reduced_100, np.average(QoI_rel_rmse_ornl_mgard_uniform_reduced_100, axis=(1)), '-^', label='Timestpe: 100, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_420, np.average(QoI_rel_rmse_ornl_mgard_uniform_reduced_420, axis=(1)), '-^', label='Timestpe: 420, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_700, np.average(QoI_rel_rmse_ornl_mgard_uniform_reduced_700, axis=(1)), '-^', label='Timestpe: 700, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_1000, np.average(QoI_rel_rmse_ornl_mgard_uniform_reduced_1000, axis=(1)), '-^', label='Timestpe: 1000, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_100, np.average(QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_100, axis=(1)), '-^', label='Timestpe: 100, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_420, np.average(QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_420, axis=(1)), '-^', label='Timestpe: 420, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_700, np.average(QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_700, axis=(1)), '-^', label='Timestpe: 700, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_1000, np.average(QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_1000, axis=(1)), '-^', label='Timestpe: 1000, MGARD (uniform) + postprocessing (PQ 8bits)')

axis.set_xlabel('Compression ratio', fontsize=14)
axis.set_ylabel('NRMSE', fontsize=15)

axis.legend(loc="lower right", bbox_to_anchor=(2.35, 0.2), fontsize=14)
#plt.subplots_adjust(wspace=0.25)
plt.suptitle('Avg QoIs',fontsize=23)


# In[23]:


# New results

# density

fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

#axis.plot(compression_ratio_mgard_uniform[:20], QoI_rel_rmse_ornl_mgard_uniform[:20,0], '-^k', label='MGARD (uniform)')
#axis.plot(compression_ratio_mgard_s_minus1[:16], QoI_rel_rmse_ornl_mgard_s_minus1[:16,0], '-^m', label='MGARD (s=-1)')
#axis.plot(compression_ratio_mgard_uniform_breg_64bits, QoI_rel_rmse_ornl_mgard_uniform_breg_64bits[:,0], '-^', label='MGARD (uniform) + postprocessing (no PQ)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_6bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_6bits[:,0], '-^', label='MGARD (uniform) + postprocessing (PQ 6bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits[:,0], '-^', label='MGARD (uniform) + postprocessing (PQ 8bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_10bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_10bits[:,0], '-^', label='MGARD (uniform) + postprocessing (PQ 10bits)')

axis.plot(compression_ratio_mgard_uniform_reduced_100, QoI_rel_rmse_ornl_mgard_uniform_reduced_100[:,0], '-^', label='Timestpe: 100, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_420, QoI_rel_rmse_ornl_mgard_uniform_reduced_420[:,0], '-^', label='Timestpe: 420, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_700, QoI_rel_rmse_ornl_mgard_uniform_reduced_700[:,0], '-^', label='Timestpe: 700, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_1000, QoI_rel_rmse_ornl_mgard_uniform_reduced_1000[:,0], '-^', label='Timestpe: 1000, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_100, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_100[:,0], '-^', label='Timestpe: 100, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_420, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_420[:,0], '-^', label='Timestpe: 420, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_700, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_700[:,0], '-^', label='Timestpe: 700, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_1000, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_1000[:,0], '-^', label='Timestpe: 1000, MGARD (uniform) + postprocessing (PQ 8bits)')


axis.set_xlabel('Compression ratio', fontsize=14)
axis.set_ylabel('NRMSE', fontsize=15)

axis.legend(loc="lower right", bbox_to_anchor=(2.35, 0.25), fontsize=14)
#plt.subplots_adjust(wspace=0.25)
plt.suptitle('QoI - Density',fontsize=23)


# In[25]:


# New results

# upara

fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

#axis.plot(compression_ratio_mgard_uniform[:20], QoI_rel_rmse_ornl_mgard_uniform[:20,1], '-^k', label='MGARD (uniform)')
#axis.plot(compression_ratio_mgard_s_minus1[:16], QoI_rel_rmse_ornl_mgard_s_minus1[:16,1], '-^m', label='MGARD (s=-1)')
#axis.plot(compression_ratio_mgard_uniform_breg_64bits, QoI_rel_rmse_ornl_mgard_uniform_breg_64bits[:,1], '-^', label='MGARD (uniform) + postprocessing (no PQ)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_6bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_6bits[:,1], '-^', label='MGARD (uniform) + postprocessing (PQ 6bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits[:,1], '-^', label='MGARD (uniform) + postprocessing (PQ 8bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_10bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_10bits[:,1], '-^', label='MGARD (uniform) + postprocessing (PQ 10bits)')

axis.plot(compression_ratio_mgard_uniform_reduced_100, QoI_rel_rmse_ornl_mgard_uniform_reduced_100[:,1], '-^', label='Timestpe: 100, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_420, QoI_rel_rmse_ornl_mgard_uniform_reduced_420[:,1], '-^', label='Timestpe: 420, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_700, QoI_rel_rmse_ornl_mgard_uniform_reduced_700[:,1], '-^', label='Timestpe: 700, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_1000, QoI_rel_rmse_ornl_mgard_uniform_reduced_1000[:,1], '-^', label='Timestpe: 1000, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_100, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_100[:,1], '-^', label='Timestpe: 100, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_420, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_420[:,1], '-^', label='Timestpe: 420, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_700, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_700[:,1], '-^', label='Timestpe: 700, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_1000, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_1000[:,1], '-^', label='Timestpe: 1000, MGARD (uniform) + postprocessing (PQ 8bits)')


axis.set_xlabel('Compression ratio', fontsize=14)
axis.set_ylabel('NRMSE', fontsize=15)

axis.legend(loc="lower right", bbox_to_anchor=(2.35, 0.25), fontsize=14)
#plt.subplots_adjust(wspace=0.25)
plt.suptitle('QoI - $U_{\mathrm{para}}$',fontsize=23)


# In[27]:


# New results

# tperp

fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

#axis.plot(compression_ratio_mgard_uniform[:20], QoI_rel_rmse_ornl_mgard_uniform[:20,2], '-^k', label='MGARD (uniform)')
#axis.plot(compression_ratio_mgard_s_minus1[:16], QoI_rel_rmse_ornl_mgard_s_minus1[:16,2], '-^m', label='MGARD (s=-1)')
#axis.plot(compression_ratio_mgard_uniform_breg_64bits, QoI_rel_rmse_ornl_mgard_uniform_breg_64bits[:,2], '-^', label='MGARD (uniform) + postprocessing (no PQ)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_6bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_6bits[:,2], '-^', label='MGARD (uniform) + postprocessing (PQ 6bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits[:,2], '-^', label='MGARD (uniform) + postprocessing (PQ 8bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_10bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_10bits[:,2], '-^', label='MGARD (uniform) + postprocessing (PQ 10bits)')

axis.plot(compression_ratio_mgard_uniform_reduced_100, QoI_rel_rmse_ornl_mgard_uniform_reduced_100[:,2], '-^', label='Timestpe: 100, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_420, QoI_rel_rmse_ornl_mgard_uniform_reduced_420[:,2], '-^', label='Timestpe: 420, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_700, QoI_rel_rmse_ornl_mgard_uniform_reduced_700[:,2], '-^', label='Timestpe: 700, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_1000, QoI_rel_rmse_ornl_mgard_uniform_reduced_1000[:,2], '-^', label='Timestpe: 1000, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_100, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_100[:,2], '-^', label='Timestpe: 100, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_420, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_420[:,2], '-^', label='Timestpe: 420, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_700, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_700[:,2], '-^', label='Timestpe: 700, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_1000, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_1000[:,2], '-^', label='Timestpe: 1000, MGARD (uniform) + postprocessing (PQ 8bits)')


axis.set_xlabel('Compression ratio', fontsize=14)
axis.set_ylabel('NRMSE', fontsize=15)

axis.legend(loc="lower right", bbox_to_anchor=(2.35, 0.25), fontsize=14)
#plt.subplots_adjust(wspace=0.25)
plt.suptitle('QoI - $T_{\mathrm{perp}}$',fontsize=23)


# In[29]:


# New results

# tpara

fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

#axis.plot(compression_ratio_mgard_uniform[:20], QoI_rel_rmse_ornl_mgard_uniform[:20,3], '-^k', label='MGARD (uniform)')
#axis.plot(compression_ratio_mgard_s_minus1[:16], QoI_rel_rmse_ornl_mgard_s_minus1[:16,3], '-^m', label='MGARD (s=-1)')
#axis.plot(compression_ratio_mgard_uniform_breg_64bits, QoI_rel_rmse_ornl_mgard_uniform_breg_64bits[:,3], '-^', label='MGARD (uniform) + postprocessing (no PQ)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_6bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_6bits[:,3], '-^', label='MGARD (uniform) + postprocessing (PQ 6bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits[:,3], '-^', label='MGARD (uniform) + postprocessing (PQ 8bits)')
#axis.plot(compression_ratio_mgard_uniform_breg_pq_10bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_10bits[:,3], '-^', label='MGARD (uniform) + postprocessing (PQ 10bits)')

axis.plot(compression_ratio_mgard_uniform_reduced_100, QoI_rel_rmse_ornl_mgard_uniform_reduced_100[:,3], '-^', label='Timestpe: 100, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_420, QoI_rel_rmse_ornl_mgard_uniform_reduced_420[:,3], '-^', label='Timestpe: 420, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_700, QoI_rel_rmse_ornl_mgard_uniform_reduced_700[:,3], '-^', label='Timestpe: 700, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_reduced_1000, QoI_rel_rmse_ornl_mgard_uniform_reduced_1000[:,3], '-^', label='Timestpe: 1000, MGARD (uniform)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_100, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_100[:,3], '-^', label='Timestpe: 100, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_420, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_420[:,3], '-^', label='Timestpe: 420, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_700, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_700[:,3], '-^', label='Timestpe: 700, MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits_1000, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits_1000[:,3], '-^', label='Timestpe: 1000, MGARD (uniform) + postprocessing (PQ 8bits)')


axis.set_xlabel('Compression ratio', fontsize=14)
axis.set_ylabel('NRMSE', fontsize=15)

axis.legend(loc="lower right", bbox_to_anchor=(2.35, 0.25), fontsize=14)
#plt.subplots_adjust(wspace=0.25)
plt.suptitle('QoI - $T_{\mathrm{para}}$',fontsize=23)


# In[43]:


# New results

# n0_avg

fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))
#axis.plot(compression_ratio_mgard_uniform[:20], QoI_rel_rmse_ornl_mgard_uniform[:20,4], '-^k', label='MGARD (uniform)')
axis.plot(compression_ratio_mgard_s_minus1[:16], QoI_rel_rmse_ornl_mgard_s_minus1[:16,4], '-^m', label='MGARD (s=-1)')
axis.plot(compression_ratio_mgard_uniform_breg_64bits, QoI_rel_rmse_ornl_mgard_uniform_breg_64bits[:,4], '-^', label='MGARD (uniform) + postprocessing (no PQ)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_6bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_6bits[:,4], '-^', label='MGARD (uniform) + postprocessing (PQ 6bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits[:,4], '-^', label='MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_10bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_10bits[:,4], '-^', label='MGARD (uniform) + postprocessing (PQ 10bits)')

axis.set_xlabel('Compression ratio', fontsize=14)
axis.set_ylabel('NRMSE', fontsize=15)

axis.legend(loc="lower right", bbox_to_anchor=(2.1, 0.25), fontsize=14)
#plt.subplots_adjust(wspace=0.25)
plt.suptitle('QoI - $n0_{\mathrm{avg}}$',fontsize=23)


# In[44]:


# New results

# T0_avg

fig, axis = plt.subplots(1,1, figsize=(6.8*1,6))

#axis.plot(compression_ratio_mgard_uniform[:20], QoI_rel_rmse_ornl_mgard_uniform[:20,5], '-^k', label='MGARD (uniform)')
axis.plot(compression_ratio_mgard_s_minus1[:16], QoI_rel_rmse_ornl_mgard_s_minus1[:16,5], '-^m', label='MGARD (s=-1)')
axis.plot(compression_ratio_mgard_uniform_breg_64bits, QoI_rel_rmse_ornl_mgard_uniform_breg_64bits[:,5], '-^', label='MGARD (uniform) + postprocessing (no PQ)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_6bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_6bits[:,5], '-^', label='MGARD (uniform) + postprocessing (PQ 6bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_8bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_8bits[:,5], '-^', label='MGARD (uniform) + postprocessing (PQ 8bits)')
axis.plot(compression_ratio_mgard_uniform_breg_pq_10bits, QoI_rel_rmse_ornl_mgard_uniform_breg_pq_10bits[:,5], '-^', label='MGARD (uniform) + postprocessing (PQ 10bits)')

axis.set_xlabel('Compression ratio', fontsize=14)
axis.set_ylabel('NRMSE', fontsize=15)

axis.legend(loc="lower right", bbox_to_anchor=(2.1, 0.25), fontsize=14)
#plt.subplots_adjust(wspace=0.25)
plt.suptitle('QoI - $T0_{\mathrm{avg}}$',fontsize=23)


# In[ ]:




