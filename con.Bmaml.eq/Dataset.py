import torch
import numpy as np
from tqdm import tqdm
import os


class AbsCosine(object):
    '''
    Abs( A Cosine(\omega x + b )) + bias + noise(std = sigma)
    '''

    def __init__(self, A = 1.0, b = 0, omega = np.pi / 10, bias = 0.5, sigma = 0.01):
        if A == None:
            A = np.random.uniform(0.5, 5.0, 1)
            b = np.random.uniform(0.1, 2.0, 1)
            omega = np.pi / np.random.uniform(10, 25, 1)

        self.A = A
        self.b = b
        self.omega = omega
        self.bias = bias
        self.sigma = sigma

    def __call__(self, i):
        return np.abs(self.A * np.cos(self.omega * i + self.b)) + self.bias + np.random.normal(0, self.sigma, 1)



class Tasks(object):

    #Alow = 0.1
    get_A = AbsCosine(A = None, bias = 0.5)
    get_b = AbsCosine(A = None, bias = 0)
    get_omega = AbsCosine(A = None, bias = 0.5)
    sigmacoe = 0.01
    xlow = -5.0
    xhigh = 5.0

    def __init__(self, shots = 10, root = './data'):
        self.shots = shots
        self.root = root
        self.train_dir = os.path.join(self.root, 'train')
        self.val_dir = os.path.join(self.root, 'val')

    def TaskLoader(self, is_train = True, time_steps = 5, total_num_of_tasks = 100):
        if is_train:
            read_dir = self.train_dir
        else:
            read_dir = self.val_dir
        while True:
            indx = np.random.randint(0, total_num_of_tasks - time_steps, size = 1)[0]
            Tasks = []
            for i in range(time_steps):
                ind = indx + i
                X = np.loadtxt(os.path.join(read_dir, 'X_{}.txt'.format(ind)))
                Y = np.loadtxt(os.path.join(read_dir, 'Y_{}.txt'.format(ind)))
                std = np.loadtxt(os.path.join(read_dir, 'std_{}.txt'.format(ind)))
                #print(X.shape, Y.shape)
                X = torch.tensor(X.astype(np.float32)[:, np.newaxis])
                Y = torch.tensor(Y.astype(np.float32)[:, np.newaxis])
                std = torch.tensor(std.astype(np.float32))
                #print(X.size())
                Sample = (X[:self.shots], Y[:self.shots], X[self.shots:], Y[self.shots:], std)
                Tasks.append(Sample)
            yield Tasks

    def TaskGenerator(self):

        now = 0
        while True:

            A = self.get_A(now)
            b = self.get_b(now)
            omega = self.get_omega(now)
            std = (self.sigmacoe * A)

            X = np.random.uniform(self.xlow, self.xhigh, self.shots * 3)
            Noise = np.random.normal(loc = 0, scale=std, size = (self.shots * 3, ))
            Y = A * np.sin(omega * X + b) + Noise

            X = X[:, np.newaxis].astype(np.float32)
            Y = Y[:, np.newaxis].astype(np.float32)
            std = std.astype(np.float32)
            Sample = (X[:self.shots], Y[:self.shots], X[self.shots:], Y[self.shots:], std)

            now = now + 1
            yield Sample

    def generate_benchmark(self, num_of_task = 1000, save_dir = './data'):
        pbar = tqdm(range(num_of_task))
        gene = self.TaskGenerator()
        train_dir = os.path.join(save_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        for i in pbar:
            task = next(gene)
            X, Y, Xtest, Ytest, std = task
            X = np.concatenate((X, Xtest))
            Y = np.concatenate((Y, Ytest))
            np.savetxt(os.path.join(train_dir, 'X_{}.txt'.format(i)), X)
            np.savetxt(os.path.join(train_dir, 'Y_{}.txt'.format(i)), Y)
            np.savetxt(os.path.join(train_dir, 'std_{}.txt'.format(i)), std)
        print('Training set done!')
        pbar = tqdm(range(100))
        val_dir = os.path.join(save_dir, 'val')
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        for i in pbar:
            task = next(gene)
            X, Y, Xtest, Ytest, std = task
            X = np.concatenate((X, Xtest))
            Y = np.concatenate((Y, Ytest))
            np.savetxt(os.path.join(val_dir, 'X_{}.txt'.format(i)), X)
            np.savetxt(os.path.join(val_dir, 'Y_{}.txt'.format(i)), Y)
            np.savetxt(os.path.join(val_dir, 'std_{}.txt'.format(i)), std)


def test():
    DL = Tasks(shots= 10)

    Ge = DL.TaskGenerator()

    for x, y, std in Ge:
        print(x.size(), y.size(), std.size())
        break

def generate_benchmark():
    DL = Tasks(shots=10)
    DL.generate_benchmark(1000, './data')

if __name__ == '__main__':
    #test()
    generate_benchmark()
