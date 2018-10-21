import torchvision.utils as vutils
import os
import torch

def save_samples(samples, filename, dir="./samples/", mean=0.5, std=0.5):
    vutils.save_image(vutils.make_grid(samples.mul(std).add(mean)),
                      os.path.join(dir,filename))

def save_checkpoint(epoch, iters, net, optim, model_name, args=None):
    torch.save({
        'epoch': epoch,
        'iterations': iters,
        'model_state': net.state_dict(),
        'optim_state': optim.state_dict(),
        'args': args
    }, './models/' + model_name + "_%d.pt"%(iters+1))

def load_checkpoint(fn):
    cp = torch.load(fn)
    return cp['epoch'], cp['iterations'], cp['model_state'], cp['optim_state']