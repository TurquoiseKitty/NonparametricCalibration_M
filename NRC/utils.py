from scipy import stats
import numpy as np
import torch
from torch.linalg import norm
from numpy.linalg import norm as npnorm
import random
import os
from sklearn.isotonic import IsotonicRegression





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


def seed_all(seed = 1234):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# code from  https://github.com/uncertainty-toolbox/uncertainty-toolbox

def iso_recal(
    exp_props: np.ndarray,
    obs_props: np.ndarray,
) -> IsotonicRegression:
    """Recalibration algorithm based on isotonic regression.
    Fits and outputs an isotonic recalibration model that maps observed
    probabilities to expected probabilities. This mapping provides
    the necessary adjustments to produce better calibrated outputs.
    Args:
        exp_props: 1D array of expected probabilities (values must span [0, 1]).
        obs_props: 1D array of observed probabilities.
    Returns:
        An sklearn IsotonicRegression recalibration model.
    """
    # Flatten
    exp_props = exp_props.flatten()
    obs_props = obs_props.flatten()

    # quants should be a rising list
    assert npnorm(np.sort(exp_props) - exp_props) < 1E-6

    iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    

    try:
        iso_model = iso_model.fit(obs_props, exp_props)
    except Exception:
        raise RuntimeError("Failed to fit isotonic regression model")

    return iso_model
