import torch
import numpy as np
from tqdm import tqdm
import os
class Tasks(object):

    #Alow = 0.1
    Alow = 1
    Ahigh = 5.0
    blow = 0.1
    bhigh = 2.0 * np.pi
    Omegalow = 0.5
    Omegahigh = 2.0
    sigmacoe = 0.01
    xlow = -5.0
    xhigh = 5.0

    def __init__(self, shots = 10, root = './data'):
        self.shots = shots
        self.root = root
        self.train_dir = os.path.join(self.root, 'train')
        self.val_dir = os.path.join(self.root, 'val')

    def TaskLoader(self, is_train = True, total_num_of_tasks = 100):
        '''
        load tasks from previously saved ones
        '''
        if is_train:
            read_dir = self.train_dir
        else:
            read_dir = self.val_dir
        while True:
            indx = np.random.randint(0, total_num_of_tasks, size = 1)[0]
            X = np.loadtxt(os.path.join(read_dir, 'X_{}.txt'.format(indx)))
            Y = np.loadtxt(os.path.join(read_dir, 'Y_{}.txt'.format(indx)))
            std = np.loadtxt(os.path.join(read_dir, 'std_{}.txt'.format(indx)))
            #print(X.shape, Y.shape)
            X = torch.tensor(X.astype(np.float32)[:, np.newaxis])
            Y = torch.tensor(Y.astype(np.float32)[:, np.newaxis])
            std = torch.tensor(std.astype(np.float32))
            #print(X.size())
            Sample = (X[:self.shots], Y[:self.shots], X[self.shots:], Y[self.shots:], std)
            yield Sample
    def TaskGenerator(self):
        '''
        Sample new tasks
        '''
        while True:
            A = np.random.uniform(self.Alow, self.Ahigh, size = 1)
            b = np.random.uniform(self.blow, self.bhigh, size = 1)
            omega = np.random.uniform(self.Omegalow, self.Omegahigh, size = 1)
            std = (self.sigmacoe * A)

            X = np.random.uniform(self.xlow, self.xhigh, self.shots * 3)
            Noise = np.random.normal(loc = 0, scale=std, size = (self.shots * 3, ))
            Y = A * np.sin(omega * X + b) + Noise

            #Sample = {'data':X,
            #          'label':Y}
            X = X[:, np.newaxis].astype(np.float32)
            Y = Y[:, np.newaxis].astype(np.float32)
            std = std.astype(np.float32)
            Sample = (X[:self.shots], Y[:self.shots], X[self.shots:], Y[self.shots:], std)

            yield Sample

    def generate_benchmark(self, num_of_task = 1000, save_dir = './data'):
        '''
        Construct task pools
        '''
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
