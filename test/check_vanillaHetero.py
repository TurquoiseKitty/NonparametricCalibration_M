from experiments.plot_utils import plot_xy_specifyBound
import numpy as np
from scipy import stats
from NRC.utils import normalZ, Upper_quant, Lower_quant, DEFAULT_mean_func, DEFAULT_layers, mu_sig_toQuants
import matplotlib.pyplot as plt
from NRC.models.DNNModels import MC_dropnet
from NRC.train.DNNTrain import DNN_Trainer
import torch


def CHECK_HNN():

    fig, (ax1, ax2) = plt.subplots(1, 2)


    X_train = np.linspace(0,15,4000)
    Y_train = DEFAULT_mean_func(X_train) + 1*np.random.randn(4000)

    X_val = np.linspace(0, 15, 1000)
    Y_val = DEFAULT_mean_func(X_val) + 1*np.random.randn(1000)

    y_pred = DEFAULT_mean_func(X_val)

    y_UP1 = y_pred + 1 * normalZ.ppf(Upper_quant)
    y_LO1 = y_pred + 1 * normalZ.ppf(Lower_quant)


    X_train = torch.Tensor(X_train).view(-1, 1).cuda()
    X_val = torch.Tensor(X_val).view(-1, 1).cuda()

    Y_train = torch.Tensor(Y_train).cuda()
    Y_val = torch.Tensor(Y_val).cuda()

    HNN_model = MC_dropnet(
        n_input = 1,
        drop_rate= 0.,
        hidden_layers= [10, 10]
    )

    DNN_Trainer(HNN_model, X_train, Y_train, X_val, Y_val,
                bat_size = 10,
                LR = 5E-3,
                N_Epoch = 100,
                early_stopping=True)


    output = HNN_model.predict(X_val)

    means = output[:, 0].detach()
    sigs = output[:, 1].detach()

    pred_quants = mu_sig_toQuants(mu = means, sig = sigs, quantiles = [Lower_quant, Upper_quant])

    pred_LO = pred_quants[0].cpu().numpy()
    pred_UP = pred_quants[1].cpu().numpy()

    plot_xy_specifyBound(
        y_pred = y_pred,

        y_UP = y_UP1,
        y_LO = y_LO1,

        y_true = Y_val.cpu().numpy(),
        x = X_val.view(-1).cpu().numpy(),
        n_subset = 300,

        ylims = [-10, 10],
        xlims = [0, 15],

        ax = ax1,
        title = "Confidence Band, Oracle"
    )

    plot_xy_specifyBound(
        y_pred = means.cpu().numpy(),

        y_UP = pred_UP,
        y_LO = pred_LO,

        y_true = Y_val.cpu().numpy(),
        x = X_val.view(-1).cpu().numpy(),
        n_subset = 300,

        ylims = [-10, 10],
        xlims = [0, 15],

        ax = ax2,
        title = "Confidence Band, MC Estimation"

    )

    plt.savefig("test/Plots_bundle/check_vanillaHetero.png")

    plt.show(block=True)





if __name__ == "__main__":
    CHECK_HNN()