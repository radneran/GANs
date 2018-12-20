#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import torch.optim as optim

from mog_eigval_dist import *
from model import *
from utils import *


# In[2]:


# 2-D dist
# target dist -- normal ([-1, 1], 0.4)
# aux dist -- normal ([1, -1], 0.4)
# input dist -- normal (0, 1)
batch_size = 512
inputs = np.random.normal(loc=[0,0], scale=[0.4,0.4], size=[batch_size, 2])
tgt = np.random.normal(loc=[-1,1], scale=[0.4,0.4], size=[batch_size, 2])
aux = np.random.normal(loc=[1,-1], scale=[0.4,0.4], size=[batch_size, 2])

out = np.vstack([tgt, aux])
kde(out[:,0], out[:,1])
kde(inputs[:,0], inputs[:,1])


# In[3]:


G = MLP_Generator(depth=4, width=16, activation=SELU,bs=512,insize=2,outsize=2).cuda()
D = MLP_Discriminator(depth=4, width=16, activation=SELU,insize=2,outsize=1).cuda()

for fc in get_modules_of_type(module_type=nn.Linear, net=G):
    selu_init(fc)
for fc in get_modules_of_type(module_type=nn.Linear, net=D):
    selu_init(fc)

g_optim = optim.RMSprop(G.parameters(), alpha=5e-4)
d_optim = optim.RMSprop(D.parameters(), alpha=5e-4)


# In[4]:


out = G(torch.FloatTensor(inputs).cuda()).detach().cpu().numpy()
kde(out[:,0], out[:,1])


# In[9]:


def train_gan(G, D, g_optim, d_optim, tmu, tsig, model_name, batch_size=512, iterations=100000, clip=0.1):
    for iteration in range(iterations):
        """L_D = -(torch.log(torch.sigmoid(D(tgt))).mean()
                + torch.log(1 - torch.sigmoid(D(fake.detach()))).mean())
        L_G = (-torch.log(torch.sigmoid(D(fake))).mean())
        """
        niters = 100 if iteration < 25 or iteration % 500 == 0 else 5
        for p in G.parameters():
            p.requires_grad = False
        for i in range(niters):
            for p in D.parameters():
                p.data.clamp_(-clip, clip)
            inputs = torch.FloatTensor(np.random.normal(loc=[0,0], scale=[0.4,0.4], size=[batch_size, 2])).cuda()
            tgt = torch.FloatTensor(np.random.normal(loc=tmu, scale=tsig, size=[batch_size, 2])).cuda()
            fake = G(inputs)
            L_D = D(tgt).mean() - D(fake.detach()).mean()
            d_optim.zero_grad()
            L_D.backward()
            d_optim.step()
        for p in G.parameters():
            p.requires_grad = True
        g_optim.zero_grad()
        L_G = - D(fake).mean()
        L_G.backward()
        g_optim.step()
        
        print('[%d/%d] Loss D: %f Loss G: %f'%(iteration, iterations, L_D, L_G))
        if iteration % 1000 == 999:
            save_checkpoint(epoch=0, iters=iteration, net=G, optim=g_optim, model_name=model_name+'_G')
            save_checkpoint(epoch=0, iters=iteration, net=D, optim=d_optim, model_name=model_name+'_D')


# In[ ]:


train_gan(G, D, g_optim, d_optim, [-1,1], [0.4,0.4], 'WMLP_toy_tgt')


# In[ ]:


out2 = G(torch.FloatTensor(inputs).cuda()).detach().cpu().numpy()
kde(out2[:,0], out2[:,1], save_file='./expt_results/MLPG_logloss_tgt')


# In[ ]:




