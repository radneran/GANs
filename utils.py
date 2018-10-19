import torchvision.utils as vutils
import os

def save_samples(samples, filename, dir="./samples/", mean=0.5, std=0.5):
    vutils.save_image(vutils.make_grid(samples.mul(std).add(mean)),
                      os.path.join(dir,filename))