import torch.optim as optim
import torchvision.utils as vutils
from Data_loader import *
from model import *

nz, bs = 32, 32
fmset, fmloader = get_cifar10(bs)
# imgs, _ = iter(fmloader).next()
# plt.imshow(vutils.make_grid(imgs).numpy()[0],cmap='gray')
# plt.show()
name = "DCGAN"
G = DCGAN_Generator().cuda()
D = DCGAN_Discriminator().cuda()
weight_init(G, 0, 0.02)
weight_init(D, 0, 0.02)
g_optim = optim.Adam(G.parameters(), 5e-5, betas=(0.5, 0.9))
d_optim = optim.Adam(D.parameters(), 5e-5, betas=(0.5, 0.9))
epochs, iterations = 10, 0
fixed_noise = torch.randn(bs, nz).cuda()

def get_loss(D, G, real, type='standard'):
    fake = G()
    if type == 'wasserstein':
        L_D = D(real).mean() - D(fake.detach()).mean()
        L_G = - D(fake).mean()
    else: # return standard loss
        L_D = -(torch.log(torch.sigmoid(D(real))).mean() \
                + torch.log(1 - torch.sigmoid(D(torch.tanh(fake).detach()))).mean())
        L_G = (-torch.log(torch.sigmoid(D(torch.tanh(fake))))).mean()
    return L_D, L_G, fake

for epoch in range(epochs):
    for data in fmloader:
        d_optim.zero_grad()
        g_optim.zero_grad()
        real, _ = data
        real = real.cuda()
        # real = data.cuda()
        # fake = G()
        L_D, L_G, _ = get_loss(D, G, real)
        L_D.backward()
        d_optim.step()
        L_G.backward()
        g_optim.step()
        iterations += 1
        print("[%d/%d][%d] L_D: %f L_G: %f" % (epoch,
                                               epochs,
                                               iterations,
                                               L_D,
                                               L_G))
        if iterations % 500 == 499:
            fake = vutils.make_grid(fake)
            vutils.save_image(fake, "./samples/{0}_{1}.png".format(name, iterations))
        if iterations % 1000 == 999:
            torch.save(G, "./models/{0}_G.pt".format(name))
            torch.save(D, "./models/{0}_D.pt".format(name))
