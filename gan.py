import torch.optim as optim
import torchvision.utils as vutils

from data_loader import *
from model import *
from utils import save_checkpoint

def get_loss(D, G, real, type='standard'):
    fake = G()
    if type == 'wasserstein':
        L_D = D(real).mean() - D(fake.detach()).mean()
        L_G = - D(fake).mean()
    else:  # return standard loss
        L_D = -(torch.log(torch.sigmoid(D(real))).mean()
                + torch.log(1 - torch.sigmoid(D(torch.tanh(fake).detach()))).mean())
        L_G = (-torch.log(torch.sigmoid(D(torch.tanh(fake))))).mean()
    return L_D, L_G, fake

def train_gan(D, G, trainloader, d_optim, g_optim, epochs, name, save_samples=False):
    iterations = 0
    for epoch in range(epochs):
        for data in trainloader:
            d_optim.zero_grad()
            g_optim.zero_grad()
            real, _ = data
            real = real.cuda()
            # real = data.cuda()
            # fake = G()
            L_D, L_G, fake = get_loss(D, G, real)
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
            if save_samples and iterations % 500 == 499:
                fake = vutils.make_grid(fake)
                vutils.save_image(fake, "./samples/{0}_{1}.png".format(name, iterations))
            if iterations % 1000 == 999:
                save_checkpoint(epoch, iterations, net=G, optim=g_optim, model_name=name + "_G")
                save_checkpoint(epoch, iterations, net=D, optim=d_optim, model_name=name + "_D")
                #torch.save(G, "./models/{0}_G.pt".format(name))
                #torch.save(D, "./models/{0}_D.pt".format(name))
    return

# taken from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=10, use_cuda=True):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def train_wgan(D, G, trainloader, d_optim, g_optim, epochs, name, save_samples=False, clip=0.01):
    iterations = 0
    for epoch in range(epochs):
        for data in trainloader:
            n_critic = 100 if iterations < 25 or iterations % 500 == 0 else 5
            for p in G.parameters():
                p.requires_grad = False
            for i in range(n_critic):
                for p in D.parameters():
                    p.data.clamp_(-clip, clip)
                d_optim.zero_grad()
                real, _ = data
                real = real.cuda()
                L_D, L_G, fake = get_loss(D, G, real, "wasserstein")
                L_D.backward()
                d_optim.step()
            for p in G.parameters():
                p.requires_grad = True
            L_D, L_G, fake = get_loss(D, G, real, "wasserstein")
            g_optim.zero_grad()
            L_G.backward()
            g_optim.step()
            iterations += 1
            print("[%d/%d][%d] L_D: %f L_G: %f" % (epoch,
                                                   epochs,
                                                   iterations,
                                                   L_D,
                                                   L_G))
            if save_samples and iterations % 500 == 499:
                fake = vutils.make_grid(fake)
                vutils.save_image(fake, "./samples/{0}_{1}.png".format(name, iterations))
            if iterations % 1000 == 999:
                torch.save(G, "./models/{0}_G.pt".format(name))
                torch.save(D, "./models/{0}_D.pt".format(name))


if __name__ == "__main__":    
    nz, bs = 32, 32
    trainset, trainloader = get_cifar10(bs)
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
