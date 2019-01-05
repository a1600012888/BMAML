import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector

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

    def __call__(self, retain_graph, paramsvec:torch.nn.ParameterList, ret_grad = True):
        pred = self.net.compute_graph(self.X, paramsvec)
        logp = self.criterion(pred, self.Y) / ((self.std ** 2) * -2.0)
        # (y - y^) **  2 / std **2

        if ret_grad:
            grad = torch.autograd.grad(logp, paramsvec, create_graph=retain_graph, #retain_graph = retain_graph,
                                       only_inputs=True)
            return grad
        return logp


def add_group(x, y, alpha = 1.0):
    # x + y*alpha
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
        '''
        retparam: kxy, dxkxy
        '''
        vecx = parameters_to_vector(x)
        vecy = parameters_to_vector(y)
        r = vecx - vecy
        r = torch.dot(r, r)
        out = torch.exp(r / ((self.sigma ** 2) * -2))
        grad = torch.autograd.grad(out, x, create_graph=retain_graph, only_inputs=True)

        return out, grad
        #group_sub = add_group(x, y, -1)
        #sum = 0
        #for a in group_sub:
        #    sum = sum + torch.dot(a.reshape(-1), a.reshape(-1))
        #out = torch.exp(sum / (self.sigma * -2))
        #grad = torch.autograd.grad(out, x, create_graph=retain_graph, only_inputs=True)
        #return out, grad
