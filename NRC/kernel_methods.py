import numpy as np
import torch
from torch.linalg import norm
from numpy.linalg import norm as npnorm
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.metrics import pairwise_distances


def kernel_estimator(
        test_Z: np.ndarray,
        recal_Z: np.ndarray,
        recal_epsilon:np.ndarray,
        quants,
        base_kernel = lambda X : np.exp(-X ** 2),
        wid = 1E-1,
):
    
    assert isinstance(test_Z, np.ndarray)
    assert isinstance(recal_Z, np.ndarray)
    assert isinstance(recal_epsilon, np.ndarray)
    
    
    assert len(test_Z.shape) == 2
    assert len(recal_Z.shape) == 2
    assert len(recal_epsilon.shape) == 1

    # quants should be a rising list
    assert npnorm(np.sort(quants) - quants) < 1E-6
    
    indices = np.argsort(recal_epsilon)
    
    sorted_epsi = recal_epsilon[indices]

    sorted_recal_Z = recal_Z[indices]
       
    dist_mat = base_kernel(pairwise_distances(test_Z/wid, sorted_recal_Z/wid))
     
    summation_matform = np.triu(np.ones((len(recal_Z), len(recal_Z))))
 
    aggregated_dist_mat = np.matmul(dist_mat, summation_matform)

    empirical_quantiles = aggregated_dist_mat / aggregated_dist_mat[:, -1:]


    quantiles_unsquze = empirical_quantiles.reshape(empirical_quantiles.shape + (-1,))

    tf_mat = quantiles_unsquze <= quants
    

    harvest_ids = np.clip(np.transpose(tf_mat.sum(axis=1), (1, 0)), a_max = len(recal_Z)-1, a_min = -1)

    return sorted_epsi[harvest_ids]          # shape (len(quants), len(test_Z))






def old_kernel_estimator(
        test_Z: torch.Tensor,
        recal_Z: torch.Tensor,
        recal_epsilon: torch.Tensor,
        quants,
        base_kernel = lambda X : torch.exp(-norm(X, dim = 2) ** 2),
        lamb = 1,
        wid = 1E-1,
):
    
    assert isinstance(test_Z, torch.Tensor)
    assert isinstance(recal_Z, torch.Tensor)
    assert isinstance(recal_epsilon, torch.Tensor)


    assert len(test_Z.shape) == 2
    assert len(recal_Z.shape) == 2
    assert len(recal_epsilon.shape) == 1

    # quants should be a rising list
    assert npnorm(np.sort(quants) - quants) < 1E-6

    quants = torch.Tensor(quants).to(test_Z.device)

    assert len(quants.shape) == 1

    
    sorted_epsi, indices = torch.sort(recal_epsilon, dim = 0)

    sorted_recal_Z = recal_Z[indices]

    test_Z_unsqueezed = test_Z.unsqueeze(1).repeat(1, len(recal_epsilon), 1)
    sorted_recal_Z_unsqueezed = sorted_recal_Z.unsqueeze(0) .repeat(len(test_Z),1,1)

    dist_mat = lamb * base_kernel((sorted_recal_Z_unsqueezed - test_Z_unsqueezed) / wid)

    summation_matform = torch.triu(torch.ones(len(recal_Z), len(recal_Z))).to(test_Z.device)
 
    aggregated_dist_mat = torch.matmul(dist_mat, summation_matform)

    empirical_quantiles = aggregated_dist_mat / aggregated_dist_mat[:, -1:]


    quantiles_unsquze = empirical_quantiles.view(empirical_quantiles.shape + (-1,))

    tf_mat = quantiles_unsquze <= quants

    harvest_ids = torch.clip(torch.permute(tf_mat.sum(dim=1), (1, 0)), max = len(recal_Z)-1)

    return sorted_epsi[harvest_ids]          # shape (len(quants), len(test_Z))



# algorithm 2 for MAQR proposed in Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification

def tau_to_quant_datasetCreate(
        Z: torch.Tensor,
        epsilon: torch.Tensor,
        quants,
        kernel = lambda X : (norm(X, dim = 2) <= 1).type(torch.float),
        wid = 1E-1
):
    tauXsample = kernel_estimator(
        test_Z = Z,
        recal_Z = Z,
        recal_epsilon = epsilon,
        quants = quants,
        base_kernel = kernel,
        lamb = 1,
        wid = wid
        )
    sample_bed = tauXsample.reshape(-1)
    quant_bed = torch.Tensor(quants).view(-1, 1).repeat(1, len(Z)).view(-1).to(Z.device)


    x_stacked = Z.repeat(len(quants), 1)
    
    train_X = torch.cat([x_stacked, quant_bed.view(-1,1)], dim=1)

    train_Y = sample_bed

    return train_X, train_Y