from scipy import stats
import numpy as np
import torch
from torch.linalg import norm
from numpy.linalg import norm as npnorm



normalZ = stats.norm(loc=0, scale = 1)


Upper_quant = 0.975
Lower_quant = 0.025


DEFAULT_mean_func = lambda x : 4*np.sin(x/15 * 2 * np.pi)
DEFAULT_hetero_sigma = lambda x : np.clip(0.2 *x *np.abs(np.sin(x)), 0.1, 2)




DEFAULT_layers = [5, 5]


def obs_vs_exp(
   y_true: torch.Tensor,
   exp_quants,
   quantile_preds: torch.Tensor
):
    assert isinstance(y_true, torch.Tensor)
    assert len(y_true.shape) == 1

    # quants should be a rising list
    assert npnorm(np.sort(exp_quants) - exp_quants) < 1E-6

    exp_quants = torch.Tensor(exp_quants).to(y_true.device)

    assert isinstance(quantile_preds, torch.Tensor)
    assert quantile_preds.shape == (len(exp_quants), len(y_true))


    tf_mat = quantile_preds >= y_true

    obs_quants = tf_mat.sum(dim= 1) / len(y_true)

    return exp_quants, obs_quants


def mu_sig_toQuants(
    mu: torch.Tensor,
    sig: torch.Tensor,
    quantiles,
):
    
    quants = torch.Tensor(np.clip(normalZ.ppf(quantiles), a_min = -5, a_max= 5)).view(-1, 1).to(mu.device)

    quant_ests = quants * sig + mu.view(1, -1).repeat(len(quantiles), 1)

    return quant_ests