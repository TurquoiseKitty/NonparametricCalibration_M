from OQR_losses import batch_interval_loss
import OQR_helper as helper
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader






            
def OQR_Trainer(
            model,
            X_train, Y_train, X_val, Y_val,
            bat_size = 128,
            LR = 1E-2,
            Decay = 1E-4,
            N_Epoch = 300,
            validate_times = 20,
            verbose = True,
            train_loss = batch_interval_loss,
            val_loss_criterias = {
                'batch_int' : batch_interval_loss,
            },
            early_stopping = True,
            patience = 10,
            monitor_name = 'batch_int',
            backdoor = None,
            harvestor = None,
            arg_corr_mult = 0.5,
            **kwargs
        ):
    

    if not harvestor:

        harvestor = {
            "training_losses": []
        }
        if early_stopping:
            harvestor["early_stopped"] = False
            harvestor["early_stopping_epoch"] = 0
            harvestor["monitor_name"] = monitor_name
            harvestor["monitor_vals"] = []

        for key in val_loss_criterias.keys():
            harvestor["val_"+key] = []

    else:

        # we are assuming that a harvestor is carefully written and carefully inserted

        assert len(harvestor["training_losses"]) == 0

        if early_stopping:

            assert "early_stopped" in harvestor.keys() and not harvestor["early_stopped"]
            assert harvestor["early_stopping_epoch"] == 0
            assert len(harvestor["monitor_vals"]) == 0

        for key in val_loss_criterias.keys():
            assert len(harvestor["val_"+key]) == 0


       


    if isinstance(X_train, np.ndarray):

        X_train, Y_train, X_val, Y_val = map(torch.Tensor, [X_train, Y_train, X_val, Y_val])

    w_tr, w_va, get_tr_weights = helper.get_wqr_weights(monitor_name, X_train, Y_train, X_val, Y_val, device = X_train.device)

    

    training_set = TensorDataset(X_train, Y_train)

    if backdoor and backdoor == "MMD_LocalTrain":
        training_loader = DataLoader(training_set, batch_size=bat_size, shuffle=False)
    else:
        training_loader = DataLoader(training_set, batch_size=bat_size, shuffle=True)

    PREV_loss = 1E5

    MIN_epoch = 0
    MIN_loss = 1E5
    # cache the state dict
    state_dict_cache = None
    q_list = torch.Tensor(np.linspace(0.01, 0.99, 100)).to(X_train.device)
    arg_corr_mult = arg_corr_mult

    if early_stopping:
        patience_count = 0

    for epoch in range(N_Epoch):
        for i_bat, (X_bat, Y_bat) in enumerate(training_loader):

            loss = model.loss(train_loss, X_bat, Y_bat, q_list,
                                batch_q=True,
                                take_step=True, args=arg_corr_mult, weights=get_tr_weights(i_bat))



        # we always want to validate
        ep_va_loss = model.update_va_loss(
            val_loss_criterias[monitor_name], X_val, Y_val, q_list,
            batch_q=True, curr_ep=epoch, num_wait=patience,
            args=arg_corr_mult, weights=w_va
        )
    

        if epoch % int(N_Epoch / validate_times) == 0:

            
            if verbose:
                print("epoch ", epoch)

                print("validation loss: ", ep_va_loss)

