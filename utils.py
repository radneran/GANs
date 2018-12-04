import torchvision.utils as vutils
import os
import torch

def save_samples(samples, filename, dir="./samples/", mean=0.5, std=0.5):
    vutils.save_image(vutils.make_grid(samples.mul(std).add(mean)),
                      os.path.join(dir,filename))

def save_checkpoint(epoch, iters, net, optim, model_name, args=None, dir='./models/'):
    torch.save({
        'epoch': epoch,
        'iterations': iters,
        'model_state': net.state_dict(),
        'optim_state': optim.state_dict(),
        'args': args
    }, os.path.join(dir, model_name + ".pt"))

def load_checkpoint(model_name, dir='./models/'):
    cp = torch.load(os.path.join(dir, model_name + ".pt"))
    return cp['epoch'], cp['iterations'], cp['model_state'], cp['optim_state'], cp['args']