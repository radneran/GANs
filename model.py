import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.upsampling import UpsamplingNearest2d

def weight_init(net, mean, std):
    for m in net.modules():
        if (isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)):
            nn.init.normal_(m.weight, mean, std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BEGAN_Generator(nn.Module):
    def __init__(self, nz=32, bs=32, size=32):
        super().__init__()
        self.h_size = size//4
        self.bs = bs
        self.nz = nz
        self.fc = nn.Linear(32, self.h_size * self.h_size * 128)
        self.conv1 = nn.Conv2d(128,128, 3, 1, 1,bias=False)
        self.conv2 = nn.Conv2d(128,128, 3, 1, 1,bias=False)
        self.conv3 = nn.Conv2d(128,64, 3, 1, 1,bias=False)
        self.conv4 = nn.Conv2d(64,64, 3, 1, 1,bias=False)
        self.conv5 = nn.Conv2d(64,32, 3, 1, 1,bias=False)
        self.conv6 = nn.Conv2d(32,32, 3, 1, 1,bias=False)
        self.conv7 = nn.Conv2d(32,3, 3, 1, 1,bias=False)
        self.ups = UpsamplingNearest2d(scale_factor=2)
    def forward(self,z=None):
        if z is None:
            z = torch.randn(self.bs,self.nz).cuda()
        z = F.elu(self.fc(z))
        z = z.view(-1, 128, self.h_size, self.h_size)
        z = self.ups(F.elu(self.conv2(F.elu(self.conv1(z)))))
        z = self.ups(F.elu(self.conv4(F.elu(self.conv3(z)))))
        z = self.conv7(F.elu(self.conv6(F.elu(self.conv5(z)))))
        return z

class Discriminator(nn.Module):
    # Encoder part of the BEGAN AE
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(3,32, 3, 1,1,bias=False)
        self.conv1 = nn.Conv2d(32,32, 3, 1,1,bias=False)
        self.conv2 = nn.Conv2d(32,64, 3, 2,1,bias=False)
        self.conv3 = nn.Conv2d(64,64, 3, 1,1,bias=False)
        self.conv4 = nn.Conv2d(64,128, 3, 2,1,bias=False)
        self.conv5 = nn.Conv2d(128,128, 3, 1,bias=False)
        self.conv6 = nn.Conv2d(128,128, 3, 1,2,bias=False)
        self.fc = nn.Linear(8 * 8 * 128, 1)

    def forward(self, x):
        out = F.elu(self.conv2(
            F.elu(self.conv1(
                F.elu(self.input(x))))))
        out = F.elu(self.conv4(F.elu(self.conv3(out))))
        out = F.elu(self.conv6(F.elu(self.conv5(out))))
        out = out.view(-1,128*8*8)
        out = self.fc(out)
        return out

class BEGAN_AutoEncoder(nn.Module):
    # AE as described in BEGAN
    def __init__(self, size=32):
        super().__init__()
        self.h_size = size//4
        self.input = nn.Conv2d(3,32, 3, 1,1,bias=False)
        self.conv1 = nn.Conv2d(32,32, 3, 1,1,bias=False)
        self.conv2 = nn.Conv2d(32,64, 3, 2,1,bias=False)
        self.conv3 = nn.Conv2d(64,64, 3, 1,1,bias=False)
        self.conv4 = nn.Conv2d(64,128, 3, 2,1,bias=False)
        self.conv5 = nn.Conv2d(128,128, 3, 1,bias=False)
        self.conv6 = nn.Conv2d(128,128, 3, 1,2,bias=False)
        self.fc1 = nn.Linear(self.h_size * self.h_size * 128, 32)
        self.fc2 = nn.Linear(32, self.h_size * self.h_size * 128)
        self.conv7 = nn.Conv2d(128,128, 3, 1, 1,bias=False)
        self.conv8 = nn.Conv2d(128,128, 3, 1, 1,bias=False)
        self.conv9 = nn.Conv2d(128,64, 3, 1, 1,bias=False)
        self.conv10 = nn.Conv2d(64,64, 3, 1, 1,bias=False)
        self.conv11 = nn.Conv2d(64,32, 3, 1, 1,bias=False)
        self.conv12 = nn.Conv2d(32,32, 3, 1, 1,bias=False)
        self.conv13 = nn.Conv2d(32,3, 3, 1, 1,bias=False)
        self.ups = UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        out = F.elu(self.conv2(F.elu(self.conv1(
            F.elu(self.input(x))))))
        out = F.elu(self.conv4(F.elu(self.conv3(out))))
        out = F.elu(self.conv6(F.elu(self.conv5(out))))
        out = out.view(-1,128*self.h_size*self.h_size)
        h = self.fc1(out)
        out = F.elu(self.fc2(h))
        out = out.view(-1, 128,self.h_size, self.h_size)
        out = self.ups(F.elu(self.conv8(F.elu(self.conv7(out)))))
        out = self.ups(F.elu(self.conv10(F.elu(self.conv9(out)))))
        out = self.conv13(F.elu(self.conv12(
            F.elu(self.conv11(out)))))
        return out

class DCGAN_Generator(nn.Module):
    def __init__(self, nz=32, bs=32):
        super().__init__()
        self.bs = bs
        self.nz = nz
        self.dconv1 = nn.ConvTranspose2d(nz,256,4,1,bias=False)
        self.dconv2 = nn.ConvTranspose2d(256,128,4,2,1,bias=False)
        self.dconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.dconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(256)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.bnorm3 = nn.BatchNorm2d(64)

    def forward(self, z=None):
        if z is None:
            z = torch.randn(self.bs,self.nz).cuda()
        z = z.view(self.bs, self.nz, 1, 1)
        out = torch.relu(self.bnorm1(self.dconv1(z)))
        out = torch.relu(self.bnorm2(self.dconv2(out)))
        out = torch.relu(self.bnorm3(self.dconv3(out)))
        out = self.dconv4(out)
        return out

class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256, 1, 4, bias=False)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.bnorm1 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = F.leaky_relu(self.bnorm1(self.conv1(x)), negative_slope=0.2)
        out = F.leaky_relu(self.bnorm2(self.conv2(out)), negative_slope=0.2)
        out = F.leaky_relu(self.bnorm3(self.conv3(out)), negative_slope=0.2)
        out = self.conv4(out)
        return out

class MLP_Generator(nn.Module):
    def __init__(self, depth, width, activation, bn=False, bs=32, insize=32, outsize=2):
        super(MLP_Generator, self).__init__()
        self.insize = insize
        self.bs = bs
        self.main = nn.Sequential()
        for i in range(depth):
            if i == 0:
                curr_layer = nn.Linear(insize, width)
            elif i == depth - 1:
                curr_layer = nn.Linear(width, outsize)
            else:
                curr_layer = nn.Linear(width, width)
            self.main.add_module("FC" + str(i), curr_layer)
            if i != depth - 1:
                if bn:
                    self.main.add_module(nn.LayerNorm(width))
                self.main.add_module("ACT" + str(i), activation())

    def forward(self, z=None):
        if z is None:
            z = torch.randn(self.bs, self.insize).cuda()
        return self.main(z)

class MLP_Discriminator(nn.Module):
    def __init__(self, depth, width, activation, bn=False, insize=2, outsize=1):
        super(MLP_Discriminator, self).__init__()
        self.main = nn.Sequential()
        for i in range(depth):
            if i == 0:
                curr_layer = nn.Linear(insize, width)
            elif i == depth - 1:
                curr_layer = nn.Linear(width, outsize)
            else:
                curr_layer = nn.Linear(width, width)
            self.main.add_module("FC" + str(i), curr_layer)
            if i != depth - 1:
                if bn:
                    self.main.add_module(nn.LayerNorm(width))
                self.main.add_module("ACT" + str(i), activation())

    def forward(self, x):
        return self.main(x)

