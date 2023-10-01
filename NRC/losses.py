# some codes from https://github.com/YoungseogChung/calibrated-quantile-uq and https://github.com/ZongxianLee/MMD_Loss.Pytorch


import torch.nn as nn
import torch
import numpy as np
from NRC.utils import normalZ, obs_vs_exp, mu_sig_toQuants

def mse_loss(predic, target):

    if len(predic.shape) != 1:
        predic = predic[:, 0]
       
    assert len(target.shape) == 1

    loss_layer = nn.MSELoss()

    return loss_layer(predic, target) 

def rmse_loss(predic, target):

    return torch.sqrt(mse_loss(predic, target))

def mean_std_norm_loss(predic, target):

    assert len(target.shape) == 1
    assert len(predic.shape) == 2

    mu = predic[:, 0]
    sigma = predic[:,1]

    loss = nn.GaussianNLLLoss()
    return loss(mu, target, sigma**2)



def mean_std_forEnsemble(predic, target):

    assert len(predic) % len(target) == 0

    n_mod = int(len(predic) / len(target))

    target = target.repeat(n_mod)

    return mean_std_norm_loss(predic, target)



def gaussian_kernel_func(
    source,
    target,
    bandwidth_list = [1, 4, 8, 16, 32]
):
    assert isinstance(source, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert len(source.shape) == 1 

    n_samples = len(source) + len(target)

    source = source.view(-1,1)
    target = target.view(-1,1)

    total = torch.cat([source, target], dim = 0)


    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_dist = ((total0 - total1)**2).sum(dim = 2)

    kernel_val =  [torch.exp(-L2_dist / wid) for wid in bandwidth_list]

    kernel_val = sum(kernel_val)

    XX = kernel_val[:len(source), :len(source)]

    XY = kernel_val[:len(source), len(source):]

    YX = kernel_val[len(source):, :len(source)]

    YY = kernel_val[len(source):, len(source):]
    	    
    loss = torch.mean(XX + YY - XY -YX)

    return loss


def MMD_Loss(source, target, bandwidth_list = [1, 4, 8, 16, 32]):


    source_len = len(source)

    with torch.no_grad():

        errs = torch.randn_like(target).to(source.device)

    real_source = source[:, 0] + errs * source[:, 1]

    loss = gaussian_kernel_func(
        real_source,
        target,
        bandwidth_list
    )

    
    return loss



def avg_pinball(y_pred_Plus_tau, Y):
    
    # follow the code given above

    assert len(Y.shape) == 1
    assert y_pred_Plus_tau.shape == (len(Y), 2)


    y_preds = y_pred_Plus_tau[:,0]
    taus = y_pred_Plus_tau[:, 1]


    diff = y_preds - Y
    with torch.no_grad():

        mask = (diff.ge(0).float() - taus).detach()

    loss = (mask * diff).mean()

    return loss


def avg_pinball_quantile(y_out, Y, q_list = np.array([0.2, 0.8])):

    # when the net predicts quantiles directly


    assert (y_out.shape == (len(Y), len(q_list))) or (y_out.shape == (len(q_list), len(Y)))

    q_list_bed =  torch.Tensor(q_list).view(-1, 1).repeat(1, len(Y)).view(-1).to(y_out.device)


    if y_out.shape == (len(y_out), len(q_list)):
        y_pred = torch.permute(y_out, (1, 0)).reshape(-1,)
    
    else:
        y_pred = y_out.reshape(-1,)

    y_pred_Plus_tau =  torch.stack([y_pred, q_list_bed],dim=1)

    return avg_pinball(y_pred_Plus_tau, Y.repeat(len(q_list)))


def avg_pinball_muSigma(y_out, Y, q_list = np.linspace(0.01,0.99,100), recal=False, recal_model=None):

    assert len(y_out) == len(Y)

    assert y_out.shape == (len(y_out), 2)

    q_list_bed =  torch.Tensor(q_list).view(-1, 1).repeat(1, len(Y)).view(-1).to(y_out.device)

    mu_bed = y_out[:, 0].repeat(len(q_list))
    sigma_bed = y_out[:, 1].repeat(len(q_list))

    if not recal:
        quant_bed = torch.Tensor(np.clip(normalZ.ppf(q_list), a_min = -1E5, a_max= 1E5)).view(-1, 1).repeat(1, len(Y)).view(-1).to(y_out.device)
    else:
        quant_bed = torch.Tensor(np.clip(normalZ.ppf(recal_model.predict(q_list)), a_min = -1E5, a_max= 1E5)).view(-1, 1).repeat(1, len(Y)).view(-1).to(y_out.device)

    y_pred = mu_bed + sigma_bed * quant_bed

    y_pred_Plus_tau =  torch.stack([y_pred, q_list_bed],dim=1)

    return avg_pinball(y_pred_Plus_tau, Y.repeat(len(q_list)))



# from paper Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification    


def BeyondPinball_loss(y_pred_Plus_tau, Y):

    # assume y_pred_Plus_tau is the predicted Q_t(Y) combined with quantile level tau
    # also assume that y_pred_Plus is produced from:

    # x_stacked = x.repeat(num_q, 1)
    # q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    # model_in = torch.cat([x_stacked, q_rep], dim=1)
    # pred_y = model(model_in)
    # y_pred_Plus_tau = torch.cat([pred_y.view(-1,1), q_rep], dim=1)

    assert len(Y.shape) == 1
    assert y_pred_Plus_tau.shape[1] == 2
    assert len(y_pred_Plus_tau) % len(Y) == 0
    

    num_pts = len(Y)
    num_q = int(len(y_pred_Plus_tau) / len(Y))

    q_rep = y_pred_Plus_tau[:,1:]

    q_list = y_pred_Plus_tau[:,1].view(num_q, num_pts)[:, 0]

    y_stacked = Y.repeat(num_q)
    y_mat = y_stacked.reshape(num_q, num_pts)


    pred_y = y_pred_Plus_tau[:, 0]

    idx_under = (y_stacked <= pred_y).reshape(num_q, num_pts)
    idx_over = ~idx_under
    coverage = torch.mean(idx_under.float(), dim=1)  # shape (num_q,)

    pred_y_mat = pred_y.reshape(num_q, num_pts)
    diff_mat = y_mat - pred_y_mat

    mean_diff_under = torch.mean(-1 * diff_mat * idx_under, dim=1)
    mean_diff_over = torch.mean(diff_mat * idx_over, dim=1)

    cov_under = coverage < q_list
    cov_over = ~cov_under
    loss_list = (cov_under * mean_diff_over) + (cov_over * mean_diff_under)

    
    loss = torch.mean(loss_list)

    return loss
 

def BeyondPinball_quantile(y_out, Y, q_list = np.array([0.2, 0.8])):

    # when the net predicts quantiles directly

    assert len(y_out) == len(Y)

    assert y_out.shape == (len(y_out), len(q_list))

    q_list_bed =  torch.Tensor(q_list).view(-1, 1).repeat(1, len(Y)).view(-1).to(y_out.device)

    y_pred = torch.permute(y_out, (1, 0)).reshape(-1,)

    y_pred_Plus_tau =  torch.stack([y_pred, q_list_bed],dim=1)

    return BeyondPinball_loss(y_pred_Plus_tau, Y)



def BeyondPinball_muSigma(y_out, Y, q_list = np.linspace(0.01,0.99,100)):

    assert len(y_out) == len(Y)

    assert y_out.shape == (len(y_out), 2)

    q_list_bed =  torch.Tensor(q_list).view(-1, 1).repeat(1, len(Y)).view(-1).to(y_out.device)

    mu_bed = y_out[:, 0].repeat(len(q_list))
    sigma_bed = y_out[:, 1].repeat(len(q_list))

    # sensitive to large quantile values. That's why we clip
    quant_bed = torch.Tensor(np.clip(normalZ.ppf(q_list), a_min = -5, a_max= 5)).view(-1, 1).repeat(1, len(Y)).view(-1).to(y_out.device)


    y_pred = mu_bed + sigma_bed * quant_bed

    y_pred_Plus_tau =  torch.stack([y_pred, q_list_bed],dim=1)

    return BeyondPinball_loss(y_pred_Plus_tau, Y)




def MACE_Loss(
    y_out,
    Y,
    q_list = np.linspace(0.01, 0.99, 100),
):


    assert y_out.shape == (len(q_list), len(Y))

    exp_quants, obs_quants = obs_vs_exp(Y, q_list, y_out)

    abs_diff_proportions = torch.abs(exp_quants - obs_quants)
    mace = torch.mean(abs_diff_proportions)

    return mace

def MACE_muSigma(
    y_out, Y, q_list = np.linspace(0.01,0.99,100), recal=False, recal_model = None
):
    assert len(y_out) == len(Y)

    assert y_out.shape == (len(y_out), 2)

    if not recal:

        y_quants = mu_sig_toQuants(y_out[:,0], y_out[:, 1], quantiles = q_list)

    else:

        y_quants = mu_sig_toQuants(y_out[:,0], y_out[:, 1], quantiles = recal_model.predict(q_list))

    return MACE_Loss(y_quants, Y, q_list)




def AGCE_Loss(
    y_out,
    Y,
    
    ratio = 0.1,
    q_list = np.linspace(0.01, 0.99, 100),
    draw_with_replacement: bool = False,
    num_trials: int = 1,
    num_group_draws: int = 10
):

    assert y_out.shape == (len(q_list), len(Y))

    num_pts = Y.shape[0]
     
    group_size = max([int(round(num_pts * ratio)), 2])
        
    score_per_trial = []  # list of worst miscalibrations encountered

    for _ in range(num_trials):
            
        group_miscal_scores = []

        for g_idx in range(num_group_draws):
            
            rand_idx = np.random.choice(
                num_pts, group_size, replace=draw_with_replacement
            )


            MACE_error = MACE_Loss(y_out[:,rand_idx], Y[rand_idx], q_list= q_list).item()

            group_miscal_scores.append(MACE_error)
                
                
        max_miscal_score = np.max(group_miscal_scores)
        score_per_trial.append(max_miscal_score)
            

    return np.mean(score_per_trial)


def AGCE_muSigma(
    y_out, Y, 
    ratio = 0.1,
    q_list = np.linspace(0.01, 0.99, 100),
    draw_with_replacement: bool = False,
    num_trials: int = 1,
    num_group_draws: int = 10,
    recal=False,
    recal_model = None
):
    assert len(y_out) == len(Y)

    assert y_out.shape == (len(y_out), 2)

    if not recal:

        y_quants = mu_sig_toQuants(y_out[:,0], y_out[:, 1], quantiles = q_list)

    else:

        y_quants = mu_sig_toQuants(y_out[:,0], y_out[:, 1], quantiles = recal_model.predict(q_list))

    return AGCE_Loss(y_quants, Y, ratio, q_list, draw_with_replacement, num_trials, num_group_draws)