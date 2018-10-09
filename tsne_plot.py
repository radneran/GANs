import os.path
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import t_sne
from sklearn.decomposition import PCA
from Data_loader import get_cifar10, get_classdata_cifar10
traincifar, _ =  get_cifar10(32)
testcifar, _ = get_cifar10(32, train=False)

if os.path.isfile('./models/testbirds.pt'):
    testbirds = torch.load('./models/testbirds.pt')
else:
    testbirds = get_classdata_cifar10(testcifar, 2)
    print(testbirds.size())
    torch.save(testbirds, './models/trainbirds.pt')

if os.path.isfile('./models/trainbirds.pt'):
    trainbirds = torch.load('./models/trainbirds.pt')
else:
    trainbirds = get_classdata_cifar10(traincifar, 2)
    print(trainbirds.size())
    torch.save(trainbirds, './models/trainbirds.pt')

pca_50 = PCA(50)
testbirds.view(-1, 3*32*32)
pca_data = pca_50.fit_transform(testbirds)
print(pca_50.explained_variance_ratio_)
plt.plot(pca_data)
plt.show()