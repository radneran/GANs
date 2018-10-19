#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.modules.upsampling import UpsamplingNearest2d


# In[2]:


bs=32
ctransform = transforms.Compose(
[transforms.Resize(32),
 transforms.ToTensor(),
 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

cifarset = torchvision.datasets.CIFAR10(root='./data',train=True,
                                       download=True, transform=ctransform)

cifarloader = torch.utils.data.DataLoader(cifarset, batch_size=bs,shuffle=True,
                                        num_workers=2)
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# In[3]:
class BAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(3,32, 3, 1,1,bias=False)
        self.conv1 = nn.Conv2d(32,32, 3, 1,1,bias=False)
        self.conv2 = nn.Conv2d(32,64, 3, 2,1,bias=False)
        self.conv3 = nn.Conv2d(64,64, 3, 1,1,bias=False)
        self.conv4 = nn.Conv2d(64,128, 3, 2,1,bias=False)
        self.conv5 = nn.Conv2d(128,128, 3, 1,bias=False)
        self.conv6 = nn.Conv2d(128,128, 3, 1,2,bias=False)
        self.fc1 = nn.Linear(8*8*128,128)
        self.fc2 = nn.Linear(128,8*8*128)
        self.conv7 = nn.Conv2d(128,128, 3, 1, 1,bias=False)
        self.conv8 = nn.Conv2d(128,128, 3, 1, 1,bias=False)
        self.conv9 = nn.Conv2d(128,64, 3, 1, 1,bias=False)
        self.conv10 = nn.Conv2d(64,64, 3, 1, 1,bias=False)
        self.conv11 = nn.Conv2d(64,32, 3, 1, 1,bias=False)
        self.conv12 = nn.Conv2d(32,32, 3, 1, 1,bias=False)
        self.conv13 = nn.Conv2d(32,3, 3, 1, 1,bias=False)
        self.ups = UpsamplingNearest2d(scale_factor=2)
    def forward(self,x,bs=32):
        x = F.elu(self.conv2(F.elu(self.conv1(F.elu(self.input(x))))))
        x = F.elu(self.conv4(F.elu(self.conv3(x))))
        x = F.elu(self.conv6(F.elu(self.conv5(x))))
        x = x.view(-1,128*8*8)
        h = self.fc1(x)
        x = self.fc2(h)
        x = x.view(-1, 128, 8, 8)
        x = self.ups(F.elu(self.conv8(F.elu(self.conv7(x)))))
        x = self.ups(F.elu(self.conv10(F.elu(self.conv9(x)))))
        x = self.conv13(F.elu(self.conv12(F.elu(self.conv11(x)))))
        return x
        
class BGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32,8*8*128)
        self.conv1 = nn.Conv2d(128,128, 3, 1, 1,bias=False)
        self.conv2 = nn.Conv2d(128,128, 3, 1, 1,bias=False)
        self.conv3 = nn.Conv2d(128,64, 3, 1, 1,bias=False)
        self.conv4 = nn.Conv2d(64,64, 3, 1, 1,bias=False)
        self.conv5 = nn.Conv2d(64,32, 3, 1, 1,bias=False)
        self.conv6 = nn.Conv2d(32,32, 3, 1, 1,bias=False)
        self.conv7 = nn.Conv2d(32,3, 3, 1, 1,bias=False)
        self.ups = UpsamplingNearest2d(scale_factor=2)
    def forward(self,z=None,nz=32,bs=32):
        if z is None:
            z = torch.randn(self.bs,nz).cuda()
        z = F.elu(self.fc(z))
        z = z.view(-1, 128, 8, 8)
        z = self.ups(F.elu(self.conv2(F.elu(self.conv1(z)))))
        z = self.ups(F.elu(self.conv4(F.elu(self.conv3(z)))))
        z = self.conv7(F.elu(self.conv6(F.elu(self.conv5(z)))))
        return z  


# In[4]:


#disc = Discriminator()
gen = BGen().cuda()
ae = BAE().cuda()
#ae = torch.load("BEGAN-ae.pt")
#gen = torch.load("BEGAN-gen.pt")
#disc.enc.weight_init(0,0.02)
#disc.dec.weight_init(0,0.02)
g_optim = optim.Adam(gen.parameters(),lr=5e-5,betas=(0.5,0.9))
ae_optim = optim.Adam(ae.parameters(),lr=5e-5,betas=(0.5,0.9))


# In[ ]:


epochs, iterations = 25, 8000
nz=32
fixed_noise = torch.randn(bs,nz).cuda()
lamda_k = 0.001
kt = 0
gamma = 0.3 # rloss_fake/rloss_real
for epoch in range(epochs):
    for idx,data in enumerate(cifarloader):
        real,_ = data
        real = real.cuda()
     #   disc.e_optim.zero_grad()
     #   disc.d_optim.zero_grad()disc.dec(disc.enc(fake))
        ae_optim.zero_grad()
        
        rloss_real = (real - ae(real)).abs().mean()
        fake = gen(torch.randn(bs,nz).cuda())
        rloss_fake = (fake.detach() - ae(fake.detach())).abs().mean()
        Ld = rloss_real - kt*rloss_fake
        Lg = (fake - ae(fake)).abs().mean()
        #loss = Ld + Lg
        #loss.backward()
        Ld.backward()
        ae_optim.step()
        g_optim.zero_grad()
        Lg.backward()
        g_optim.step()
        with torch.no_grad():
            gd_bal = (gamma*rloss_real.detach() - rloss_fake.detach())
            kt += lamda_k* gd_bal
            kt = max(min(1, kt), 0)
            M_g = rloss_real.detach() + torch.abs(gd_bal)
        iterations+=1
        print("[%d/%d][%d] M_g: %f R_loss_real: %f R_loss_fake: %f D_loss: %f G_loss: %f kt: %f"%(epoch,epochs, iterations,M_g,rloss_real,
                                                                              rloss_fake,
                                                                             Ld,
                                                                             Lg,kt))
        if iterations % 500 == 0:
            fake = gen(fixed_noise)
            #recon_fake = disc.dec(disc.enc(fake))
            #recon_real = disc.dec(disc.enc(real))
            recon_fake = ae(fake)
            recon_real = ae(real)
            samples = torch.cat((real,recon_real,fake,recon_fake))
            samples = vutils.make_grid(samples.mul(0.5).add(0.5))
            vutils.save_image(samples,"./samples/BEGAN-{0}.png".format(iterations))


# In[ ]:

# In[ ]:


#with torch.no_grad():
    #data,_ = iter(cifarloader).next()
    #h = disc.enc(data.cuda())
    #r = disc.dec(h)
#print(h.size(), r.size()
 #    )


# In[ ]:




