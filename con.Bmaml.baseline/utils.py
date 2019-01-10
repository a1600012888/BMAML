import math

class AvgMeter(object):
    name = 'No name'
    def __init__(self, name = 'No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0
    def update(self, mean_var, count = 1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


class PolyLearningRatePolicy(object):
    #_default_poly = 0.8

    def __init__(self, lr = 0.1, max_iter = 100000.0, poly = 0.8):
        self.max_iter = float(max_iter)
        self.lr = lr
        self.poly = poly

    def __call__(self, itr):
        a = 1 - itr / self.max_iter

        lr = self.lr * (a ** self.poly)

        return lr