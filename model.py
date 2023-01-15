import torch
import numpy as np
from metrics import cal_clustering_metric
from sklearn.cluster import KMeans
import tools

class GAE(torch.nn.Module):
    def __init__(self, X, X2, labels, layers=None, increase_neighbor=1, num_neighbors=3, learning_rate=10**-3,
                 max_iter=50, max_epoch=10, A=None, update=True, m=10, B=None):
        super(GAE, self).__init__()
        if layers is None:
            layers = [1024, 256, 64]
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.X = X
        self.X2 = X2
        self.labels = labels
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.A = A
        self.num_neighbors = num_neighbors
        self.embedding_dim = layers[-1]
        self.mid_dim = layers[1]
        self.input_dim = layers[0]
        self.update = update
        self.centroids = torch.zeros(int(m), self.embedding_dim)
        self.m = m
        self.increase_neighbors = increase_neighbor
        self.B = B
        self.training = True
        self.max_neighbors = self.cal_max_neighbors()

        self.embedding = None
        self._build_up()

    def cal_max_neighbors(self):
        size = self.X.shape[0]
        num_clusters = np.unique(self.labels).shape[0]
        max_neighbors = 5 * self.increase_neighbors
        if(max_neighbors==0):
            max_neighbors = 3
        return max_neighbors

    def get_Laplacian_from_A2(self, A):
        A = A + torch.eye(A.size(0))
        degree = torch.sum(A, dim=1).pow(-0.5).diag()
        return degree.mm(A).mm(degree)

    def get_Laplacian_from_A(self, A):
        A = A + torch.eye(A.size(0))
        degree = torch.sum(A, dim=1).pow(-0.5)
        return (A * degree).t() * degree

    def _build_up(self):
        self.W1 = self.get_weight_initial([self.input_dim, self.mid_dim])
        self.W2 = self.get_weight_initial([self.mid_dim, self.embedding_dim])

    def get_weight_initial(self, shape):
        bound = np.sqrt(2.0 / shape[0]*(1+0.01))
        ini = torch.rand(shape) * 2 * bound - bound
        return torch.nn.Parameter(ini, requires_grad=True)

    def clustering(self, B, k_means=True):
        n_clusters = np.unique(self.labels).shape[0]
        if k_means:
            embedding = self.embedding.cpu().detach().numpy()
            km = KMeans(n_clusters=n_clusters).fit(embedding)
            prediction = km.predict(embedding)
            acc, nmi = cal_clustering_metric(self.labels, prediction)
            print('k-means --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi))
        else:
            degreeB = torch.sum(B, dim=0).pow(-0.5).diag()
            B = B.matmul(degreeB)
            try:
                vectors, _, _ = torch.svd(B)
                indicator = vectors[:, 0:n_clusters]
            except:
                print('svd can convergence')
                return
            indicator = indicator / (indicator.norm(dim=1)+10**-10).repeat(n_clusters, 1).t()
            indicator = indicator.cpu().numpy()
            km = KMeans(n_clusters=n_clusters).fit(indicator)
            prediction = km.predict(indicator)
            acc, nmi = cal_clustering_metric(self.labels, prediction)
            # print('SC --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi))
            return acc, nmi

    def distance(self, X, Y, square=True):
        """
        Compute Euclidean distances between two sets of samples
        Basic framework: pytorch
        :param X: d * n, where d is dimensions and n is number of data points in X
        :param Y: d * m, where m is number of data points in Y
        :param square: whether distances are squared, default value is True
        :return: n * m, distance matrix
        """
        n = X.shape[1]
        m = Y.shape[1]
        x = torch.norm(X, dim=0)
        x = x * x
        x = torch.t(x.repeat(m, 1))

        y = torch.norm(Y, dim=0)
        y = y * y
        y = y.repeat(n, 1)
        crossing_term = torch.t(X).matmul(Y)
        result = x + y - 2 * crossing_term
        result = result.relu()
        if not square:
            result = torch.sqrt(result)
        result = torch.max(result, result.t())
        return result

    def forward(self):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.X = self.X.cuda()
        self.X2 = self.X2.cuda()
        B = self.B.to(device)
        degreeB = torch.sum(B, dim=0).pow(-1).diag()
        embedding = B.matmul(degreeB.matmul(B.t().matmul(self.X.matmul(self.W1))))
        embedding = torch.nn.functional.relu(embedding)
        self.embedding = B.matmul(degreeB.matmul(B.t().matmul(embedding.matmul(self.W2))))
        Dt = degreeB.matmul(B.t().matmul(B)).sum(dim=1).pow(-0.5).diag()
        LA = Dt.matmul(degreeB.matmul(torch.t(B).matmul(B.matmul(Dt)))).to(device)
        embedding2 = LA.matmul(self.X2.matmul(self.W1))
        embedding2 = torch.nn.functional.relu(embedding2)
        embedding2 = LA.matmul(embedding2.matmul(self.W2))
        f = tools.distance2(self.embedding.t(), embedding2.t(), square=False)
        recons_b = torch.nn.functional.softmax(-f)
        return recons_b

    def build_loss2(self, recons_B, B):
        loss = 0
        loss -= B * torch.log(recons_B + 10**-10)
        loss = loss.sum(dim=1)
        loss = loss.mean()
        return loss

    def reconstruct_B(self):
        centroids = torch.tensor(self.centroids).cuda()
        f = tools.distance2(self.embedding.t(), centroids.t(), square=False)
        f = torch.tensor(f)
        B = tools.getB_via_CAN(f, self.num_neighbors)
        return B

    def recons_c(self):
        tmpcentroids = torch.zeros(self.m, self.embedding_dim)
        self.B = self.B.cpu()
        Bsum = self.B.sum(dim=0)
        embedding = self.embedding.cpu()
        for j in range(self.m):
            tmp2 = Bsum[j]
            tmp = self.B[:, j].reshape(self.B.shape[0], 1).repeat(1, self.embedding_dim)
            tmp1 = (tmp * embedding).sum(dim=0)
            if (tmp2 > 0):
                tmpcentroids[j, :] = tmp1 / tmp2
        return tmpcentroids

    def run(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        for epoch in range(self.max_epoch):
            for i in range(self.max_iter):
                optimizer.zero_grad()
                recons_b = self()
                loss = self.build_loss2(recons_b.cuda(), self.B.cuda())
                loss.backward()
                optimizer.step()
            scheduler.step()
            iterNum = 25
            self.num_neighbors += self.increase_neighbors
            if self.num_neighbors > self.max_neighbors:
                self.num_neighbors = int(self.max_neighbors)
            tmpB = self.B
            while (iterNum > 0):
                self.centroids = tools.recons_c2(self.m, tmpB.cuda(), self.embedding, self.embedding_dim)
                regularization = torch.full(self.centroids.size(), 10 ** -10)
                self.centroids = self.centroids + regularization.cuda()
                tmpB = tools.reconstruct_B(self.centroids, self.embedding, self.num_neighbors)
                iterNum = iterNum - 1
            self.X2 = tools.recons_c2(self.m, tmpB.cuda(), self.X, self.X.shape[1])
            self.B = tmpB
        acc, nmi = self.clustering(self.B, k_means=False)
        return acc, nmi