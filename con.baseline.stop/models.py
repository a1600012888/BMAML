import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from typing import List, Tuple


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

    def compute_graph(self, x, paramsvec:torch.nn.ParameterList):
        y = x
        for i in range(0, len(paramsvec), 2):
            #print('req grad: ', paramsvec[i].requires_grad, paramsvec[i+1].requires_grad)
            y = F.linear(y, weight=paramsvec[i], bias=paramsvec[i + 1])
            if i < 6:
                y = F.relu(y)
        return y


class RnnForWeight(nn.Module):
    def __init__(self, input_size, hidden_size = 50, num_layers = 1, dropout = 0):
        # LSTM input shape: (seq, batch, inp)
        super(RnnForWeight, self).__init__()
        #self.LSTM = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.LSTM = LSTM(input_size, hidden_size, batch_first=False)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, Ms:List[torch.nn.ParameterList]):
        vecs = []
        for paramsvec in Ms:
            vec = parameters_to_vector(paramsvec)
            vec = vec.view(1, (vec.numel()))
            vecs.append(vec)
        #inp = torch.stack(vecs, 0)    # this line of code will lead to a unbinded gradient!!!
        inp = vecs

        hx = torch.zeros((1, self.hidden_size)).to(vecs[0].device)
        cx = torch.zeros((1, self.hidden_size)).to(vecs[0].device)
        initial_state = (hx, cx)

        out_hidden, end_state = self.LSTM(inp, initial_state)

        out = torch.squeeze(out_hidden)
        return out


class LSTM(nn.Module):
    '''
    input shape:  (seq, batch, feature)
    '''

    def __init__(self, input_size, hidden_size, batch_first=False):
        """Initialize params."""
        #super(PersonaLSTMAttentionDot, self).__init__()
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, state:Tuple[torch.Tensor, torch.Tensor]):#, ctx, ctx_mask=None):
        """Propogate input through the network."""
        # tag = None  #
        def recurrence(input, state:Tuple[torch.Tensor, torch.Tensor]):
            """Recurrence helper."""
            hx, cx = state  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)  # o_t
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        #steps = range(input.size(0))
        steps = range(len(input))
        for i in steps:
            state = recurrence(input[i], state)
            #print(input.size())
            #print('h', state[0].size())
            if isinstance(state, tuple):
                output.append(state[0])
            else:
                output.append(state)

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        # Unbind gradient!!!
        # output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        # if self.batch_first:
        #     output = output.transpose(0, 1)

        out = output[-1]
        return out, state






def test():
    net = ThreeLayer()
    paramsvec = net.params2vec()
    for i in range(len(paramsvec)):
        print(paramsvec[i].mean())
        paramsvec[i] = torch.nn.Parameter(torch.zeros_like(paramsvec[i]))
        print(paramsvec[i].mean())

    net.load_from_vecs(paramsvec)
    paramsvec = net.params2vec()
    for i in range(len(paramsvec)):
        print(paramsvec[i].mean())


def test_rnn():
    net = ThreeLayer()
    paramsvec = net.params2vec()
    num_inp = parameters_to_vector(paramsvec).numel()

    rnn = RnnForWeight(num_inp)

    for p in rnn.parameters():
        p.requires_grad = False

    M = []
    M.append(paramsvec)
    for i in range(0):
        m = torch.nn.ParameterList([torch.nn.Parameter(torch.randn_like(pa) * 0.01) for pa in paramsvec])
        M.append(m)

    out = rnn(M)
    print(out.size())

    loss = torch.sum(out)
    print('before', M[-1][0].requires_grad)
    grad = torch.autograd.grad(loss, M[-1][0], create_graph=True,  only_inputs=True)
    print('af', M[-1][0].requires_grad)
    #print(grad)
    new_loss = torch.sum(grad[0])
    print(new_loss)

    #print(new_loss, M[0][0].grad)
    new_loss.backward()
    #grad = torch.autograd.grad(new_loss, M[-1][0], create_graph=False, only_inputs=True)
    #new_loss.backward()


if __name__ == '__main__':
    #test()
    test_rnn()

#nn.Module.named_parameters()
