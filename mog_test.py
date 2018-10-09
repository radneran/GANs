from mog_eigval_dist import *
from model import *


batch_size = 512
#sigma = 0.01
#mus = np.vstack([np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(batch_size))
#x_real = mus + sigma * torch.randn([batch_size, 2])
#kde(x_real.numpy()[:,0], x_real.numpy()[:,1], bbox=(-2,2,-2,2), save_file="./expt_results/target_mog.png")

G = MLP_Generator(4, 16, nn.ReLU, bs=batch_size).cuda()
D = MLP_Discriminator(4, 16, nn.ReLU).cuda()
x_out = G().cpu().detach().numpy()
kde(x_out[:,0], x_out[:,1], bbox=(-2,2,-2,2))

# Experiments to run
# Three different tests: MOG convergence, Re and Im eigv,
# Vector space i.e. streamline plot ard eqm
# How do gradient regularization as in WGAN-GP & Gradient-reg GAN
# differ in how they affect the resulting vector space
# Factors to test
# Loss function: Wasserstien (W), W + Numerics, W + GP, W + Reg
# GAN loss, GAN + Numerics, GAN + GP, GAN + Reg
#