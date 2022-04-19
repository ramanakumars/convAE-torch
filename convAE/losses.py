from .model import *
import torch.nn.functional as F

def mse_loss(y_true, y_pred):
    mse = torch.mean(torch.sum(torch.square(y_true - y_pred), axis=(-1, -2)), axis=1)
    return torch.mean(mse)

def kl_loss(mu, sig, mup, sig0=-4):
    kl = 0.5*torch.mean(-1 - sig + sig0 + (torch.square(mu-mup) + torch.exp(sig))/np.exp(sig0), axis=-1)
    return torch.mean(kl)

def target_distribution(q):
    weight = q ** 2 / torch.sum(q, axis=0)
    return  (weight.t() / torch.sum(weight, axis=1)).t()

def dec_loss(q):
    p = target_distribution(q)

    return -nn.KLDivLoss(reduction='mean')(q, p)
    #return torch.mean(torch.sum(p*(p.log() - q.log()), axis=1))

