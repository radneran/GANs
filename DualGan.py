
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from utils import *

# In[2]:

nz=100
bs=32
#only for square images
class Critic(nn.Module):
    def __init__(self, imsize, cpg = 16):
        super(Critic, self).__init__()
        self.main = nn.Sequential()
        self.add_modules(imsize, cpg)
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(mean=0,std=0.02)
        #for m in self.modules():
            #if not isinstance(m,(Critic,nn.Sequential)):
            #    torch.nn.init.normal_(m.weight,mean=0,std=0.02)
            #print(m)
            
            
    def add_modules(self, imsize, cpg, c_in=3):
        layer_num = 0
        num_c = 64
        while imsize > 4:
            self.main.add_module("pyramid_conv{0}".format(layer_num),nn.Conv2d(c_in, num_c, 4, 2, 1, bias=False))
            self.main.add_module("pyramid_norm{0}".format(layer_num),nn.GroupNorm(num_c//cpg, num_c))
            self.main.add_module("pyramid_act{0}".format(layer_num),nn.LeakyReLU())
            layer_num+=1
            c_in=num_c
            num_c*=2
            imsize = (imsize-2)/2 + 1
        self.main.add_module("out_conv{0}".format(layer_num),nn.Conv2d(c_in, 1, 4, bias=False))
        
    def forward(self, x, mean=True):
        x = self.main(x)
        return x.mean(0) if mean else x
# Generator is U-net (AE w skip connections)
class UGenerator(nn.Module):
    def __init__(self):
        super(UGenerator, self).__init__()
        #=============== Down-sampling =================
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128,4,2,1,bias=False)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,4,2,1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,256,4,bias=False)
        #=============== Up-sampling =================
        self.dconv1 = nn.ConvTranspose2d(256, 256, 4, 1,bias=False)
        self.dbnorm1 = nn.BatchNorm2d(256)
        self.dconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1,bias=False)
        self.dbnorm2 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.dbnorm3 = nn.BatchNorm2d(64)
        self.dconv4 = nn.ConvTranspose2d(64, 3, 4,2,1, bias=False)
        
        for m in self.modules():
            if not isinstance(m,UGenerator):
                if isinstance(m, nn.BatchNorm2d):
                    torch.nn.init.normal_(m.weight,mean=1,std=0.02)
                    torch.nn.init.constant_(m.bias,0)
                torch.nn.init.normal_(m.weight,mean=0,std=0.02)
            print(m)
                
    def forward(self,x):
        x1 = F.relu(self.bnorm1(self.conv1(x)))
        x2 = F.relu(self.bnorm2(self.conv2(x1)))
        x3 = F.relu(self.bnorm3(self.conv3(x2)))
        x4 = F.relu(self.bnorm3(self.conv4(x3)))
        x5 = F.relu(self.dbnorm1(self.dconv1(x4)))
        x6 = F.relu(self.dbnorm2(self.dconv2(x5))) + x2
        x7 = F.relu(self.dbnorm3(self.dconv3(x6))) + x1
        out = torch.tanh(self.dconv4(x7))
        return out

class Generator(nn.Module):
    def __init__(self,bs=32):
        super(Generator, self).__init__()        
        self.bs = bs
        #Orig parameters
        self.dconv2 = nn.ConvTranspose2d(nz, 256, 4, 1,bias=False)
        self.bnorm2 = nn.BatchNorm2d(256)
        self.dconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1,bias=False)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.dconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(64)
        self.dconv5 = nn.ConvTranspose2d(64, 3, 4,2,1, bias=False)

        for m in self.modules():
            if not isinstance(m,Generator):
                if isinstance(m, nn.BatchNorm2d):
                    torch.nn.init.normal_(m.weight,mean=1,std=0.02)
                    torch.nn.init.constant_(m.bias,0)
                torch.nn.init.normal_(m.weight,mean=0,std=0.02)
            print(m)

    def forward(self, z=None,nz=100):
        if z is None:
            z = torch.randn(self.bs,nz).cuda()
        z = z.view(self.bs, nz, 1, 1)
        z = F.relu(self.bnorm2(self.dconv2(z)))
        z = F.relu(self.bnorm3(self.dconv3(z)))
        z = F.relu(self.bnorm4(self.dconv4(z)))
        z = torch.tanh(self.dconv5(z))
        return z

class WGAN(nn.Module):
    def __init__(self,imsize, gp=True, bs=32,nz=100,u=False,load_cp=False, model_name="WGAN"):
        super(WGAN, self).__init__()
        self.critic = Critic(imsize).cuda()
        self.gen = UGenerator() if u else Generator(bs=bs).cuda()
        self.gp = gp
        self.bs=bs
        self.model_name = model_name
        if gp:
            self.g_optim = optim.Adam(self.gen.parameters(), lr=1e-4,betas=(0.5,0.999))
            self.c_optim = optim.Adam(self.critic.parameters(), lr=1e-4,betas=(0.5,0.999))
        else:
            self.g_optim = optim.RMSprop(self.gen.parameters(), lr=5e-5).cuda()
            self.c_optim = optim.RMSprop(self.critic.parameters(), lr=5e-5).cuda()
        if not u:
            self.noise = torch.FloatTensor(bs,nz,1,1).cuda()
        if load_cp:
            _, _, critic_sd, c_optim_sd, _ = load_checkpoint(self.model_name + "_D")
            _, self.iterations, gen_sd, g_optim_sd, _ = load_checkpoint(self.model_name + "_G")
            self.critic.load_state_dict(critic_sd)
            self.gen.load_state_dict(gen_sd)
            self.c_optim.load_state_dict(c_optim_sd)
            self.g_optim.load_state_dict(g_optim_sd)
            
        self.input = torch.FloatTensor().cuda()
    
    def critic_loss(self, real, fake):
        #fake = self.gen(z=z)
        C_fake = self.critic(fake)
        C_real = self.critic(real)
        W_dist = C_real - C_fake
        if self.gp:
            grad_penalty = self.get_grad_penalty(real,fake)
            return -W_dist + grad_penalty, W_dist
        return -W_dist, W_dist
    
    def gen_loss(self, z):
        return -self.critic(self.gen(z))
        
    def get_grad_penalty(self, real, fake, lamda=10):
        bs = real.size()[0]
        alpha = torch.rand(bs,1)
        alpha = alpha.expand(bs, real.nelement()//bs).contiguous().view(real.size()).cuda()
       # print(alpha.shape, real.shape)
        interpol = (alpha * real + (1 - alpha) * fake).cuda()
        c_interpol = self.critic(interpol, mean=False)
        grads = torch.autograd.grad(inputs=interpol, outputs=c_interpol,
                                    create_graph=True,retain_graph=True,
                                    grad_outputs=torch.ones(c_interpol.size()).cuda(),
                                    only_inputs=True
                                    )[0]
        grads = grads.view(grads.size()[0], -1)
        return lamda * torch.mean((grads.norm(2, dim=1) - 1)**2)
    
    def get_samples(loader, dataiter):
        if not dataiter:
            dataiter = iter(loader)
            data = dataiter.next()
            return dataiter, data
        data = next(dataiter, None)
        if data is None:
            dataiter = iter(loader)
            data = dataiter.next()
        return dataiter, data
    
    def get_z(self, zloader, ziter):
        return get_samples(zloader, ziter) if zloader else None, self.noise.resize_(self.bs,
                                                                                    nz,
                                                                                    1,
                                                                                    1).normal_()
    
    def toggle_grad(self,net,force=None):    
        for p in net.parameters():
            if force is not None:
                p.requires_grad = force
            p.requires_grad = False if p.requires_grad else True
    
    def train(self, trainloader, epochs, lamda=10, zloader=None, save_samps=True):
        iterations = 0 if not self.iterations else self.iterations
        fixed_noise=torch.randn(self.bs, 100).cuda()
        for ep in range(epochs):
            dataiter = iter(trainloader)
            data = next(dataiter, None)
            ziter = None
            while data is not None:
                c_iter = 100 if iterations < 25 or iterations % 500 == 0 else 5
                self.toggle_grad(self.gen,force=False)
                #self.toggle_grad(self.critic,force=True)
                for i in range(c_iter):
                    self.c_optim.zero_grad()
                    real,_= (item.cuda() for item in data)
                    ziter, z = self.get_z(zloader, ziter)
                    self.input.resize_as_(real).copy_(real)
                    fake = self.gen(z)
                    if real.size()[0] < self.bs:
                        fake = fake[0:real.size()[0]]
                    c_loss, W_dist = self.critic_loss(self.input, fake)
                    c_loss.backward()
                    self.c_optim.step()
                    data = next(dataiter, None)
                    if not data or data[0].size(0) < 32: break
                self.toggle_grad(self.gen)
                #self.toggle_grad(self.critic)
                ziter, z = self.get_z(zloader,ziter)
                self.g_optim.zero_grad()
                g_loss = self.gen_loss(z)
                g_loss.backward()
                self.g_optim.step()
                iterations+=1
                print("[%d/%d][%d] Wdist: %f C_loss: %f G_loss: %f"%(ep,epochs,iterations,
                                                                    W_dist,c_loss,g_loss))
                if save_samps and iterations % 500 == 0:
                    save_samples(self.gen(fixed_noise).cpu(), "{0}_{1}.png".format(self.model_name, iterations))
                if iterations % 1000 == 0:
                    save_checkpoint(ep, iterations, self.gen, self.g_optim, self.model_name+"_G")
                    save_checkpoint(ep, iterations, self.critic, self.c_optim, self.model_name+"_D")

# Take real values to embed in image
#class EncoderNet(nn.Module):
    #def __init__
class DualGAN():
    def __init__(self,imsize=32,gp=True):
        self.critic_a = Critic(imsize).cuda()
        self.critic_b = Critic(imsize).cuda()
        #ab, ba: ab takes samples from domain a to gen domain b vice-versa
        self.gen_ab = UGenerator().cuda()
        #self.gen_ab = Generator().cuda()
        self.gen_ba = UGenerator().cuda()
        self.gp=gp
        """if load:
            torch.load('models/ca_{0}.pt'.format(fname))
            torch.load('models/cb_{0}.pt'.format(fname))
            torch.load('models/gba_{0}.pt'.format(fname))
            torch.load('models/gab_{0}.pt'.format(fname))"""
        if gp:
            self.gab_optim = optim.Adam(self.gen_ab.parameters(), lr=1e-4,betas=(0.5,0.999))
            self.gba_optim = optim.Adam(self.gen_ba.parameters(), lr=1e-4,betas=(0.5,0.999))
            self.ca_optim = optim.Adam(self.critic_a.parameters(), lr=1e-4,betas=(0.5,0.999))
            self.cb_optim = optim.Adam(self.critic_b.parameters(), lr=1e-4,betas=(0.5,0.999))
        else:
            self.gab_optim = optim.RMSprop(self.gen_ab.parameters(), lr=5e-5)
            self.ca_optim = optim.RMSprop(self.critic_a.parameters(), lr=5e-5)
            self.gba_optim = optim.RMSprop(self.gen_ba.parameters(), lr=5e-5)
            self.cb_optim = optim.RMSprop(self.critic_b.parameters(), lr=5e-5)
        print(self.critic_a, self.critic_b)
   
    def critic_loss(self,critic, real, fake):
        C_fake = critic(fake)
        C_real = critic(real)
        W_dist = C_real - C_fake
        if self.gp:
            grad_penalty = self.get_grad_penalty(critic,real,fake)
            return -W_dist + grad_penalty, W_dist
        return -W_dist, W_dist
    
    def recon_loss(self,real, recon):
        return torch.mean(torch.norm(real - recon))
    
    def gen_loss(self, realA, realB, lamdaA,lamdaB):
        genA = self.gen_ba(realB)
        genB = self.gen_ab(realA)
        reconA = self.gen_ba(genB)
        reconB = self.gen_ab(genA)
        rlossA = self.recon_loss(realA, reconA)
        rlossB = self.recon_loss(realB, reconB)
        g_loss = (lamdaA * rlossA
                  + lamdaB * rlossB
                 - self.critic_a(genA) - self.critic_b(genB))
        return g_loss, rlossA, rlossB
    
    def get_grad_penalty(self,critic, real, fake, lamda=10):
        bs = real.size()[0]
        alpha = torch.rand(bs,1)
        alpha = alpha.expand(bs, real.nelement()//bs).contiguous().view(real.size()).cuda()
        
        interpol = (alpha * real + (1 - alpha) * fake).cuda()
        c_interpol = critic(interpol, mean=False)
        grads = torch.autograd.grad(inputs=interpol, outputs=c_interpol,
                                    create_graph=True,retain_graph=True,
                                    grad_outputs=torch.ones(c_interpol.size()).cuda(),
                                    only_inputs=True
                                    )[0]
        grads = grads.view(grads.size()[0], -1)
        return lamda * torch.mean((grads.norm(2, dim=1) - 1)**2)
    
    def toggle_grad(self,net,force=None):    
        for p in net.parameters():
            if force is not None:
                p.requires_grad = force
            p.requires_grad = False if p.requires_grad else True
            
    def train(self,aloader, bloader, epochs, fname, lamdaA=100, lamdaB=100, bs=16, load=False):
        if load:
            self.critic_a = (torch.load('models/ca_{0}.pt'.format(fname)))
            self.critic_b = (torch.load('models/cb_{0}.pt'.format(fname)))
            self.gen_ba = (torch.load('models/gba_{0}.pt'.format(fname)))
            self.gen_ba = (torch.load('models/gab_{0}.pt'.format(fname)))
        iterations = 0
        for ep in range(epochs):
            aiter = iter(aloader)
            biter = iter(bloader)
            adata = next(aiter, None)
            bdata = next(biter, None)
            fixedrealA,_ = adata
            fixedrealB,_ = bdata
            vutils.save_image(torchvision.utils.make_grid(torch.cat((fixedrealA,fixedrealB))).mul(0.5).add(0.5),
                          '{0}/fixed_real_samplesAB.png'.format("samples"))
            while adata and bdata:
                c_iter = 100 if iterations < 25 or iterations % 500 == 0 else 5
                """self.toggle_grad(self.gen_ab,force=False)
                self.toggle_grad(self.gen_ba,force=False)
                self.toggle_grad(self.critic_a,force=True)
                self.toggle_grad(self.critic_b,force=True)"""
                for i in range(c_iter):
                    self.ca_optim.zero_grad()
                    self.cb_optim.zero_grad()
                    realA,_ = (item.cuda() for item in adata)
                    realB,_ = (item.cuda() for item in bdata)
                    genA = self.gen_ba(realB)
                    genB = self.gen_ab(realA)
                    c_lossA, W_distA = self.critic_loss(self.critic_a,realA, genA)
                    c_lossB, W_distB = self.critic_loss(self.critic_b,realB, genB)
                    c_lossA.backward(retain_graph=True)
                    c_lossB.backward()
                    self.ca_optim.step()
                    self.cb_optim.step()
                    adata = next(aiter, None)
                    bdata = next(biter, None)
                    if not adata or not bdata: break
                if not adata or not bdata: break
                """self.toggle_grad(self.gen_ab)
                self.toggle_grad(self.gen_ba)
                self.toggle_grad(self.critic_a)
                self.toggle_grad(self.critic_b)"""
                self.gab_optim.zero_grad()
                self.gba_optim.zero_grad()
                g_loss, rlossA, rlossB = self.gen_loss(realA, realB, lamdaA, lamdaB)
                g_loss.backward()
                self.gab_optim.step()
                self.gba_optim.step()
                iterations+=1
                print("[%d/%d][%d] lossA: %f WdistA: %f lossB: %f WdistB: %f\n                 lossG: %f rlossA: %f rlossB: %f"%(ep,epochs,iterations,c_lossA,W_distA,
                                                 c_lossB,W_distB,g_loss,rlossA,rlossB))
                if iterations % 500 == 0:
                    genA = self.gen_ba(fixedrealB.cuda())
                    genB = self.gen_ab(fixedrealA.cuda())
                    reconA = self.gen_ba(genB)
                    reconB = self.gen_ab(genA)
                    genA = genA.data.mul(0.5).add(0.5)
                    genB = genB.data.mul(0.5).add(0.5)
                    reconA = reconA.data.mul(0.5).add(0.5)
                    reconB = reconB.data.mul(0.5).add(0.5)
                    cat = torchvision.utils.make_grid(torch.cat((genA,genB,reconA,reconB)))
                    vutils.save_image(cat,
                          '{0}/{1}_samples_{2}.png'.format("samples",fname,iterations))
                if iterations % 1000 == 999:
                    torch.save(self.critic_a, 'models/ca_{0}.pt'.format(fname))
                    torch.save(self.critic_b, 'models/cb_{0}.pt'.format(fname))
                    torch.save(self.gen_ba, 'models/gba_{0}.pt'.format(fname))
                    torch.save(self.gen_ab, 'models/gab_{0}.pt'.format(fname))
    
