from config import logger

import torch
from torch import nn, Tensor
import copy

class Sine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.register_buffer('w0', torch.tensor(w0))

    def forward(self, input):
        return torch.sin(self.w0 * input)

@torch.no_grad()
def first_layer_sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            #to-do: see why this initialization
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class TMLPLoss(nn.Module):
    def __init__(self, n_layers, *args):
        super().__init__()

        assert(n_layers == len(args))
        self.n_layers = n_layers

        for idx, val in enumerate(args, 0):
            setattr(self, f'lambda{idx}', val)


    def forward(self, pred, target):
        tot = 0.0
        assert(self.n_layers == len(pred))
        for idx in range(self.n_layers):
            tot += getattr(self, f'lambda{idx}') * torch.mean((pred[idx] - target) ** 2)

        return tot

class TMLP(nn.Module):
    def __init__(self, 
                 in_features=3, 
                 out_features=1, 
                 num_hidden_layers=5, 
                 hidden_features=64,
                 dropout=0.0):
        super().__init__()

        logger.debug("TLMP initialization")
        self.out_features = out_features
        self.first_layer_init = None
        self.poly = 2
        self.dropout = dropout

        #hidden features
        num_hidden_features = hidden_features
        hidden_features = []
        for _ in range(num_hidden_layers + 1):
            hidden_features.append(num_hidden_features)

        logger.debug(f'num layers: {len(hidden_features)}')
        logger.debug(f'net size: {hidden_features}')

        #net
        #activation function sine
        nl = Sine()
        self.weight_init = first_layer_sine_init

        self.net = []
        #Layer(0)
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features[0]), nl))
        setattr(self, 'out_layer0', nn.Linear(num_hidden_features, out_features))
        getattr(self, 'out_layer0').apply(self.weight_init)

        #hidden layers(i)
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features[i], hidden_features[i+1]), copy.deepcopy(nl)))
            setattr(self, f'out_layer{i+1}', nn.Linear(num_hidden_features, out_features * self.poly))
            getattr(self, f'out_layer{i+1}').apply(self.weight_init)

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.weight_init)

    def forward(self, coords, layers=None):
        feature = coords
        preds = []

        #use it for the training 
        if layers is None:
            for i, module in enumerate(self.net):
                feature = module(feature) #[N][hidden_size]
                residual = getattr(self, f'out_layer{i}')(feature)#l=0[N][1],l>1[N][2]
                residual = residual.reshape(*residual.shape[0:-1], self.out_features, -1)#l=0[N][1][1],l>1[N][1][2]
                residual = torch.prod(residual, dim=-1)#ti = ti1 Hadamard product ti2 [N][1]
                preds.append(residual)
            return preds
        else:
            helper_layers = layers
            for i, module in enumerate(self.net):
                feature = module(feature)
                residual = getattr(self, f'out_layer{i}')(feature)
                residual = residual.view(*residual.shape[0:-1], self.out_featueres, -1)
                residual = torch.prod(residual, dim=-1)#ti = ti1 Hadamard product ti2 
                preds.append(residual)
                mask = helper_layers > len(preds)
                feature = feature[mask]
                helper_layers = helper_layers[mask]
                if helper_layers.sum() == 0:
                    break
            return preds