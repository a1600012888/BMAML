import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from models import RnnForWeight

class LogP(object):

    def __init__(self, net, criterion):
        self.criterion = criterion
        self.net = net
    def update(self, X, Y, std, net = None):
        self.X = X
        self.Y = Y
        self.std = std
        if net is not None:
            self.net = net

    def __call__(self, retain_graph, paramsvec, ret_grad = True):
        pred = self.net.compute_graph(self.X, paramsvec)
        logp = self.criterion(pred, self.Y) / ((self.std ** 2) * -2.0)

        if ret_grad:
            grad = torch.autograd.grad(logp, paramsvec, create_graph=retain_graph, #retain_graph = retain_graph,
                                       only_inputs=True)
            return grad
        return logp


def add_group(x, y, alpha = 1.0):
    ret = []
    for i in range(len(x)):
        ret.append(x[i] + y[i] * alpha)
    return ret

def divide_group(x, div):
    assert div != 0, 'dividing zero!!'

    ret = []
    for i in range(len(x)):
        ret.append(x[i] / div)
    return ret



class RBFKernelOnWeights(object):
    def __init__(self, sigma = 1.0):
        self.sigma = sigma

    def __call__(self, x, y, retain_graph = False):
        vecx = parameters_to_vector(x)
        vecy = parameters_to_vector(y)
        r = vecx - vecy
        r = torch.dot(r, r)
        out = torch.exp(r / (self.sigma * -2))
        grad = torch.autograd.grad(out, x, create_graph=retain_graph, only_inputs=True)

        #print(grad)
        return out, grad
        #group_sub = add_group(x, y, -1)
        #sum = 0
        #for a in group_sub:
        #    sum = sum + torch.dot(a.reshape(-1), a.reshape(-1))
        #out = torch.exp(sum / (self.sigma * -2))
        #grad = torch.autograd.grad(out, x, create_graph=retain_graph, only_inputs=True)
        #return out, grad


class RecurrentRBFKernelOnWeights(object):
    def __init__(self, input_size, sigma = 1.0):

        self.sigma = sigma * 2
        self.RnnfWeights = RnnForWeight(input_size = input_size, hidden_size = 50,
                                       num_layers = 1, dropout = 0)


    def __call__(self, xs, ys, retain_graph = False):
        x = xs[-1]
        #print('xxx', x)

        xfeature = self.RnnfWeights(xs)
        yfeature = self.RnnfWeights(ys)
        #xfeature = parameters_to_vector(xs[-1])
        #yfeature = parameters_to_vector(ys[-1])

        r = xfeature - yfeature
        r = torch.dot(r, r)
        out = torch.exp(r / (self.sigma * -2))
        grad = torch.autograd.grad(out, x, create_graph=retain_graph, only_inputs=True)

        #print(out.requires_grad, grad[0].requires_grad)
        return out, grad
