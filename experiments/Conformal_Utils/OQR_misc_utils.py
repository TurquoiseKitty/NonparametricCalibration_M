"""
Code copied from https://github.com/YoungseogChung/calibrated-quantile-uq
"""
import tqdm
import random
import math
import numpy as np
from scipy.interpolate import interp1d
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from numpy import histogramdd


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_q_idx(exp_props, q):
    target_idx = None
    for idx, x in enumerate(exp_props):
        if idx + 1 == exp_props.shape[0]:
            if round(q, 2) == round(float(exp_props[-1]), 2):
                target_idx = exp_props.shape[0] - 1
            break
        if x <= q < exp_props[idx + 1]:
            target_idx = idx
            break
    if target_idx is None:
        import pdb; pdb.set_trace()
        raise ValueError('q must be within exp_props')
    return target_idx


def gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    """
    Evaluate loss_fn for eqch q in q_listKBinsDiscretizer
    loss_fn must only take in a scalar q
    """
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss


def discretize_domain(x_arr, batch_size):
    num_pts, dim_x = x_arr.shape

    group_list = []
    for d in range(dim_x):
        dim_order = np.argsort(x_arr[:, d]).flatten()
        curr_group = [dim_order[i:i + batch_size] for i in
                      range(0, num_pts, batch_size)]
        assert len(curr_group) == math.ceil(num_pts / batch_size)
        group_list.append(curr_group)
    return group_list


def discretize_domain_old(x_arr, min_pts):
    num_pts, dim_x = x_arr.shape
    # n_bins = 2 * np.ones(dim_x).astype(int)

    group_data_idxs = []
    while len(group_data_idxs) < 1:
        n_bins = np.random.randint(low=1, high=3, size=dim_x)
        H, edges = histogramdd(x_arr, bins=n_bins)

        group_idxs = np.where(H >= min_pts)
        group_bounds = []
        for g_idx in zip(*group_idxs):
            group_bounds.append([(edges[i][x], edges[i][x+1])
                                 for i, x in enumerate(g_idx)])

        for b_list in group_bounds:
            good_dim_idxs = []
            for d_idx, (l, u) in enumerate(b_list):
                good_dim_idxs.append((l <= x_arr[:, d_idx]) * (x_arr[:, d_idx] < u))
            curr_group_1 = np.prod(np.stack(good_dim_idxs, axis=0), axis=0)
            curr_group_idx = np.where(curr_group_1.flatten() > 0)
            if curr_group_idx[0].size < min_pts:
                continue
            group_data_idxs.append(curr_group_idx[0])

    rand_dim_idx = np.random.randint(dim_x)
    group_pts_idx = list(np.concatenate(group_data_idxs))

    if num_pts - len(group_pts_idx) >= (min_pts//2):
        rest_rand_sorted = list(np.argsort(x_arr[:, rand_dim_idx]))
        for item in group_pts_idx:
            rest_rand_sorted.remove(item)
        rest_group_size = int(min_pts//2)
        beg_idx = 0
        rest_group_data_idxs = []
        while beg_idx < len(rest_rand_sorted):
            if beg_idx + rest_group_size >= len(rest_rand_sorted):
                end_idx = len(rest_rand_sorted)
            else:
                end_idx = beg_idx + rest_group_size

            if np.array(rest_rand_sorted[beg_idx:end_idx]).size >= rest_group_size:
                rest_group_data_idxs.append(np.array(rest_rand_sorted[beg_idx:end_idx]))
            beg_idx = end_idx
        group_data_idxs.extend(rest_group_data_idxs)

    assert np.array([x.size > 0 for x in group_data_idxs]).all()

    return group_data_idxs




if __name__ == '__main__':
    temp_x = np.random.uniform(0, 100, size=[100, 2])
    # num_bins, idxs = discretize_domain(temp_x)

    group_idxs = discretize_domain(temp_x, 30)
    cum_num_pts = 0
    for i in group_idxs:
        g = temp_x[i.flatten()]
        print(g.shape)
        cum_num_pts += g.shape[0]
        plt.plot(g[:,0], g[:,1], 'o')
    print(cum_num_pts)
    plt.show()

    # for i in idxs:
    #     g = temp_x[i.flatten()]
    #     print(g.shape)
    #     cum_num_pts += g.shape[0]
    #     plt.plot(g[:,0], g[:,1], '^')
    # plt.show()
    # assert cum_num_pts == temp_x.shape[0]
    # for i in range(5):
    #     for j in range(5):
    #
    # for i in range(num_bins):
    #     plt.plot(temp_x[idxs[i], 0], temp_x[idxs[i], 1], 'o')
    # plt.show()


