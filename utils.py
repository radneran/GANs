import torchvision.utils as vutils
import os

def save_samples(samples, filename, dir="./samples/", mean=0, var=1):
    vutils.save_image(vutils.make_grid(samples.add(1).mul(0.5)),
                      os.path.join(dir,filename))