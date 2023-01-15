import torch
import data_loader as loader
import warnings
import random
import tools
import argparse
from model import GAE

def main(args):
    warnings.filterwarnings("ignore")
    if args.datasetName == 'mnist_test':
        [data, labels] = loader.load_MNIST_Test()
    else:
        [data, labels] = loader.load_data(args.datasetName)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    X = torch.Tensor(data)
    m = args.AnchorNum
    k = args.k0
    increase_neighbor = args.increase_k

    input_dim = data.shape[1]
    layers = None
    layers = [input_dim, 256, 32]
    randomNum = random.sample(range(0, X.shape[0]), m)
    centroids = X[randomNum, :]
    regularization = torch.full(centroids.size(), 10 ** -10)
    centroids = centroids+regularization
    iterNum = 25
    while (iterNum > 0):
        f = tools.distance2(X.t(), centroids.t(), square=False)
        B = tools.getB_via_CAN(f, k)
        centroids = tools.recons_c2(m, B, X, X.shape[1])
        iterNum = iterNum - 1
    gae = GAE(X, centroids, labels, layers=layers, num_neighbors=k, increase_neighbor=increase_neighbor, max_iter=200, max_epoch=5,
              update=True, learning_rate=5 * 10 ** -2, m=m, B=B)
    gae.to(device)
    ACC, NMI = gae.run()

    print('SC --- ACC: %5.4f, NMI: %5.4f' % (ACC, NMI))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AnchorGAE')

    parser.add_argument('--AnchorNum', type=int, default=400,
                        help='Initialize the number of anchors.')
    parser.add_argument('--k0', type=int, default=3,
                        help='Initialize the k-sparsity.')
    parser.add_argument('--increase_k', type=int, default=6,
                        help='Initialize the size of the self-increasing sparsity.')
    parser.add_argument('--datasetName', type=str, default='usps_all',
                        help='The name of the dataset')
    args = parser.parse_args()

    main(args)