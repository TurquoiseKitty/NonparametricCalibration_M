from NRC.losses import mse_loss, rmse_loss, mean_std_norm_loss
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch

def LSTM_Trainer(
        model,
        X_train_raw,
        Y_train_raw,
        X_val_raw,
        Y_val_raw,
        bat_size = 128,
        LR = 1E-2,
        Decay = 1E-4,
        N_Epoch = 1000,
        validate_gaprate = 1/20,
        
        loss_criteria = mse_loss,
        val_loss_criterias = {
            "rmse": rmse_loss
        },
        verbose = True,
        early_stopping = False,
        patience = 10,
        monitor_name = "rmse"
        
    ):
    

    X_train, Y_train, X_val, Y_val = X_train_raw, Y_train_raw, X_val_raw, Y_val_raw


    if isinstance(X_train, np.ndarray):

        X_train, Y_train, X_val, Y_val = map(torch.Tensor, [X_train, Y_train, X_val, Y_val])
    
    X_train, Y_train, X_val, Y_val = X_train.to(model.device), Y_train.to(model.device), X_val.to(model.device), Y_val.to(model.device)


    training_set = TensorDataset(X_train, Y_train)
    training_loader = DataLoader(training_set, batch_size=bat_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr = LR, weight_decay=Decay)


    PREV_loss = 1E7
    patience_count = 0


    for epoch in range(N_Epoch):
        for i_bat, (X_bat, Y_bat) in enumerate(training_loader):


            optimizer.zero_grad()

            hidden = model.init_hidden(len(X_bat))
    
            cell = model.init_cell(len(X_bat))

            loss = loss_criteria(model(X_bat, hidden, cell)[0], Y_bat)

            loss.backward()

            optimizer.step()


        hidden = model.init_hidden(len(X_val))
    
        cell = model.init_cell(len(X_val))
        val_output = model(X_val, hidden, cell)[0]



        if early_stopping:
            patience_val_loss = val_loss_criterias[monitor_name](val_output, Y_val).item()
            if patience_val_loss > PREV_loss:
                patience_count += 1
            
            PREV_loss = patience_val_loss

        
        if early_stopping and patience_count >= patience and verbose:

            print("Early Stopped at Epoch ", epoch)
            break





        if epoch % int(N_Epoch * validate_gaprate) == 0 and verbose:

            

            
            print("epoch ", epoch)
            # print("prediction samples:")
            # print("input: ")
            # print(X_val[:5].detach())
            # print("output: ")
            # print(val_output[:5].detach())
            for name in val_loss_criterias.keys():

                val_loss = val_loss_criterias[name](val_output, Y_val).item()


                print("     loss: {0}, {1}".format(name, val_loss))
    

def Ensemble_LSTM_Trainer(self,
            X_train_raw,
            Y_train_raw,
            X_val_raw,
            Y_val_raw,
            bat_size = 128,
            LR = 1E-2,
            Decay = 1E-4,
            N_Epoch = 1000,
            validate_gaprate = 1/20,
            
            loss_criteria = mean_std_norm_loss,
            val_loss_criterias = {
                "rmse": rmse_loss
            },
            verbose = True,
            early_stopping = False,
            patience = 10,
            monitor_name = "rmse"
            
        ):
        

        X_train, Y_train, X_val, Y_val = X_train_raw, Y_train_raw, X_val_raw, Y_val_raw


        if isinstance(X_train, np.ndarray):

            X_train, Y_train, X_val, Y_val = map(torch.Tensor, [X_train, Y_train, X_val, Y_val])
        
        X_train, Y_train, X_val, Y_val = X_train.to(self.device), Y_train.to(self.device), X_val.to(self.device), Y_val.to(self.device)


        training_set = TensorDataset(X_train, Y_train)
        training_loader = DataLoader(training_set, batch_size=bat_size, shuffle=True)

        # optimizer list
        optimizer_list = []

        for i in range(self.n_models):

            optimizer_list.append(optim.Adam(self.ensembles[i].parameters(), lr = LR, weight_decay=Decay))


        PREV_loss = 1E7
        patience_count = 0


        for epoch in range(N_Epoch):
            for i_bat, (X_bat, Y_bat) in enumerate(training_loader):

                for i in range(self.n_models):


                    optimizer_list[i].zero_grad()

                    hidden = self.ensembles[i].init_hidden(len(X_bat))
        
                    cell = self.ensembles[i].init_cell(len(X_bat))

                    loss = loss_criteria(self.ensembles[i](X_bat, hidden, cell)[0], Y_bat)

                    loss.backward()

                    optimizer_list[i].step()


            val_output = self.predict(X_val)



            if early_stopping:
                patience_val_loss = val_loss_criterias[monitor_name](val_output, Y_val).item()
                if patience_val_loss > PREV_loss:
                    patience_count += 1
                
                PREV_loss = patience_val_loss

            
            if early_stopping and patience_count >= patience and verbose:

                print("Early Stopped at Epoch ", epoch)
                break





            if epoch % int(N_Epoch * validate_gaprate) == 0 and verbose:

                

                
                print("epoch ", epoch)
                # print("prediction samples:")
                # print("input: ")
                # print(X_val[:5].detach())
                # print("output: ")
                # print(val_output[:5].detach())
                for name in val_loss_criterias.keys():

                    val_loss = val_loss_criterias[name](val_output, Y_val).item()


                    print("     loss: {0}, {1}".format(name, val_loss))



