import scipy.io as scio
import numpy as np
import torch
from torchvision import datasets, transforms

MNIST_10 = 'MnistData_10_uni'
MNIST = 'mnist_all'
fashionMNIST = 'fashionMNIST_full'
USPS = 'usps_all'
segment = 'segment_uni'
Isolet = 'Isolet'

def load_data(name):
    path = './data/{}.mat'.format(name)
    data = scio.loadmat(path)
    if(name==USPS):
        labels = data[USPS][:, 256]
        X = data[USPS][:, 0:256]
    elif (name == fashionMNIST):
        labels = data['labels_full']
        X = data['data_full']
    else:
        labels = data['Y']
        X = data['X']
    labels = np.reshape(labels, (labels.shape[0],))
    X = X.astype(np.float)
    X /= np.max(X)
    return X, labels

def load_MNIST_Test():
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,), (0.3081,))
        ])), shuffle=True)
    data = test_loader.dataset
    labels = data.test_labels.detach().numpy()
    labels = labels.astype(np.uint8)
    labels = np.reshape(labels, (labels.shape[0],))
    X = data.test_data
    X = X.detach().numpy()
    X = X.astype(np.float)
    X /= np.max(X)
    X = X.reshape(-1, 784)
    return X, labels