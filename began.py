from torch.optim import Adam

from data_loader import *
from model import *
from utils import *

# hyper-params
proportional_gain = 0.001  # lambda_k
diversity_ratio = 0.3  # gamma
lr_generator = 1e-4
lr_discriminator = 1e-4
batch_size = 32
z_dim = 32

trainset, trainloader = get_cifar10(batch_size)
D = BEGAN_AutoEncoder().cuda()
G = BEGAN_Generator().cuda()
balancer = 0  # k
weight_init(D, 0, 0.02)
weight_init(G, 0, 0.02)
g_optim = Adam(G.parameters(), lr=lr_generator, betas=(0.9, 0.999))
d_optim = Adam(D.parameters(), lr=lr_discriminator, betas=(0.9, 0.999))

epochs, iterations = 10, 0
fixed_noise = torch.randn(batch_size, z_dim).cuda()


# BEGAN maintains equilibrium between the losses of D and G using proportional
# control theory. The balancer variable controls how much emphasis is placed on
# the D(G()) in calculating D's loss. Balancer is updated using a feedback loop at
# each training step such that it maintains the equality L_D/L_G = diversity_ratio.
# proportional_gain is how much the balancer is updated at each step.

def autoencoder_loss(D, x, p=1):
    # lp distance between x and D(x)
    return (((x - D(x)).abs()**p).sum()**(1/p))/(x.size()[0]) # meaned over batchsize
    #return (x - D(x)).abs().mean()

def began_loss(D, G, real):
    global balancer
    D_real = autoencoder_loss(D, real)
    L_D = D_real - balancer * autoencoder_loss(D, G().detach())
    L_G = D_gen = autoencoder_loss(D, G())
    return L_D, L_G, D_real, D_gen


for epoch in range(epochs):
    for data in trainloader:
        real, _ = data
        real = real.cuda()
        g_optim.zero_grad()
        d_optim.zero_grad()
        L_D, L_G, D_real, D_gen = began_loss(D, G, real)
        L_D.backward()
        L_G.backward()
        d_optim.step()
        g_optim.step()
        # update balancer
        current_balance = diversity_ratio * D_real.detach() - D_gen.detach()
        balancer += proportional_gain * current_balance
        balancer = max(0, min(1, balancer))  # 0 <= k <= 1
        convergence_measure = D_real + torch.abs(current_balance)
        iterations += 1
        print("[%d/%d][%d] M_global: %f L_D: %f L_G: %f D_real: %f k_t: %f" % (epoch,
                                                                    epochs,
                                                                    iterations,
                                                                    convergence_measure,
                                                                    L_D, L_G, D_real,
                                                                    balancer))
        if iterations % 500 == 499:
            samples = G(fixed_noise)
            save_samples(samples, "BEGAN_{0}.png".format(iterations))
        if iterations % 1000 == 999:
            torch.save(D, "./models/BEGAN_D.pt")
            torch.save(G, "./models/BEGAN_G.pt")
