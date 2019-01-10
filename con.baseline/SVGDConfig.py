from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Tuple, List, Dict
import torch
from config import add_group, divide_group


class SteinVariationalGradientDescentBase(object):
    '''
    This SVGD is modified for a recurrent kernel
    '''


    @abstractmethod
    def Kernel(self, x:torch.Tensor, y:torch.Tensor,
               retain_graph = False) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Positive-definite Kernel
        Comupte K(x,y) and dxK(x,y)
        Arguments:
        x    -- (torch.Tensor)
        y    -- (torch.Tensor)
        rparam  -- (Kxy: torch.Tensor, dxKxy: torch.Tensor)
        '''
        raise NotImplementedError
        pass

    @abstractmethod
    def NablaLogP(self, retain_graph, x:torch.Tensor) -> torch.Tensor:

        '''
        nabla_x logp(x)
        Caculate gradient of logp(x) with respect to x
        Arguments:
        x    -- (torch.Tensor)
        '''
        raise NotImplementedError
        pass

    def InitMomentumUpdaters(self, num = None):
        if num is not None:
            self.momentum_updaters = []
            for i in range(num):
                self.momentum_updaters.append(MomentumRunningMean(0.9))
        else:
            for m in self.momentum_updaters:
                m.reset()

    def step(self, Xs:List[torch.Tensor], step_size = 1e-3, retain_graph = False) -> List[torch.Tensor]:
        '''
        Perform a single step of SVGD
        Arguments
        param X:    -- (List[torch.Tensor]) list of particles
        rparam: X': list of particles abtained by a single step of SVGD
        '''

        X = Xs[-1]
        Xss = [ [Xs[j][i] for j in range(len(Xs))] for i in range(len(Xs[0])) ]
        #print(Xss)
        #print('lll', len(Xss), len(Xss[0]))
        M = len(X)
        Grads = []
        Ret = []

        dxlogps = []
        for i in range(M):
            xi = X[i]
            dxlogps.append(self.NablaLogP(retain_graph, xi))
        for i in range(M): # This can be optimized by torch.nn.PairwiseDistance
            xi = X[i]
            #xi.retain_grad()
            gradi = [torch.zeros_like(xii) for xii in xi]
            for j in range(M):
                #xj = X[j]
                dxlogp = dxlogps[j]
                #dxlogp = self.NablaLogP(retain_graph, xj)
                #kxy, dxkxy = self.Kernel(xj, xi, retain_graph)
                kxy, dxkxy = self.Kernel(Xss[j], Xss[i], retain_graph)
                #print(dxkxy)
                gradi = add_group(gradi, dxlogp, kxy /(1.0 * M))   # divide by M for normalize
                gradi = add_group(gradi, dxkxy, 1.0/(1.0 * M))
                #gradi = gradi + kxy * self.NablaLogP(retain_graph, xj)
                #gradi = gradi + dxkxy

            #gradi = gradi / M
            Grads.append(gradi)

        for i in range(M):
            #X[i] = add_group(X[i], Grads[i], step_size)
            #print(Grads[i])
            Grads[i] = self.momentum_updaters[i](Grads[i])
            #print(Grads[i][0])
            Ret.append(add_group(X[i], Grads[i], step_size))
            #X[i] = X[i] + Grads[i] * step_size
        return Ret

class MomentumRunningMean(object):
    '''
    history <- history + momentum * now
    normalize <- normalize + momentum * 1

    ret: history / normalize

    '''
    def __init__(self, momentum = 0.9):
        self.momentum = momentum
        self.reset()
    def reset(self):
        self.history = None
        self.normalize = 0
    def __call__(self, now, count = 1):
        if self.history is None:
            self.history = [torch.zeros_like(xx) for xx in now]
        self.history = add_group(now, self.history, self.momentum)
        self.normalize = self.normalize * self.momentum + 1
        update = divide_group(self.history, self.normalize)
        return update


class NablaMaker(object):
    def __init__(self, func):
        assert callable(func)
        self.func = func

    def __call__(self, retain_graph, *args, **kwargs):
        inp = args[0]
        inp.detach_()  #..  If detach, no gradient computing!
        #inp.retain_grad()
        inp.requires_grad = True
        out = self.func(*args, **kwargs)
        grad = torch.autograd.grad(out, inp, retain_graph=retain_graph, only_inputs=True)[0]
        return grad


class LogPGaussian(object):
    def __init__(self, mu:torch.Tensor = 0, sigma:torch.tensor = 1):
        self.mu = mu
        self.sigma = sigma
        print('mumu', mu)

    def __call__(self, x):
        r = x - self.mu
        r = torch.dot(r, r)
        return r / (self.sigma * -2)




def test():
    class SteinVariationalGradientDescent(SteinVariationalGradientDescentBase):
        Kernel = RBFKernel(sigma=5.0)
        NablaLogP = NablaMaker(LogPGaussian(0, 1))

    LogP = LogPGaussian(0, 1)
    SVGD = SteinVariationalGradientDescent()

    x = torch.ones((7))
    y = torch.zeros((7))
    k  =SVGD.Kernel(x, y)
    logp = LogP(x)
    print(x,y)
    dxLogP = SVGD.NablaLogP(x)
    print(x, y)
    print(logp, dxLogP)
    print(x.size(), k.size(), logp.size(),  dxLogP.size())

    x, y = SVGD.step([x, y])

if __name__ == '__main__':
    test()
