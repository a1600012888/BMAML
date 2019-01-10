import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayer(nn.Module):

    def __init__(self):
        super(ThreeLayer, self).__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=40)
        self.fc3 = nn.Linear(in_features=40, out_features=40)
        self.fc4 = nn.Linear(in_features=40, out_features=1)

    def forward(self, x):
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        y = F.relu(y)
        y = self.fc4(y)
        return y

    def params2vec(self):
        paramsvec = torch.nn.ParameterList(self.parameters())
        self.state_dict()
        return paramsvec

    def load_from_vecs(self, paramsvec):
        dic = self.state_dict()
        for i, key in enumerate(dic.keys()):
            dic[key] = paramsvec[i]
        self.load_state_dict(dic)

    def compute_graph(self, x, paramsvec):
        y = x
        for i in range(0, len(paramsvec), 2):
            #print('req grad: ', paramsvec[i].requires_grad, paramsvec[i+1].requires_grad)
            y = F.linear(y, weight=paramsvec[i], bias=paramsvec[i + 1])
            if i < 6:
                y = F.relu(y)
        return y


class RnnForNet(nn.Module):
    def __init__(self, paramsvec, hidden_size = 50, out_size = 30):
        super(RnnForNet, self).__init__()
        self.linears = []
        for p in paramsvec:
            self.linears.append(nn.Linear(torch.numel(p), hidden_size))



def test():
    net = ThreeLayer()
    paramsvec = net.para2vec()
    for i in range(len(paramsvec)):
        print(paramsvec[i].mean())
        paramsvec[i] = torch.nn.Parameter(torch.zeros_like(paramsvec[i]))
        print(paramsvec[i].mean())

    net.load_from_vecs(paramsvec)
    paramsvec = net.para2vec()
    for i in range(len(paramsvec)):
        print(paramsvec[i].mean())


if __name__ == '__main__':
    test()

#nn.Module.named_parameters()
