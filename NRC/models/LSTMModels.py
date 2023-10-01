
import torch
import torch.nn as nn
import copy
import numpy as np





class Vanilla_LSTM(nn.Module):
    def __init__(self,
                 COV_DIM,
                 LSTM_HIDDEN_DIM,
                 LSTM_LAYER = 1,
                 normalize = False,
                 output_scheme = "mu_sigma",
                 device = torch.device("cuda")):

        super(Vanilla_LSTM, self).__init__()

        self.COV_DIM = COV_DIM
        self.LSTM_LAYER = LSTM_LAYER
        self.LSTM_HIDDEN_DIM = LSTM_HIDDEN_DIM
        
        self.lstm = nn.LSTM(input_size=COV_DIM,
                            hidden_size=LSTM_HIDDEN_DIM,
                            num_layers=LSTM_LAYER,
                            bias=True,
                            batch_first=True).to(device)
        
        self.normalize = normalize

        self.device = device
        
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()

        # output. can be specified arbitrarily
        self.output_scheme = output_scheme
        if output_scheme == "pred":
            self.pred = nn.Linear(LSTM_HIDDEN_DIM * LSTM_LAYER, 1).to(device)

        elif output_scheme == "mu_sigma":

            self.distribution_mu = nn.Linear(LSTM_HIDDEN_DIM * LSTM_LAYER, 1).to(device)
            self.distribution_presigma = nn.Linear(LSTM_HIDDEN_DIM * LSTM_LAYER, 1).to(device)
            self.distribution_sigma = nn.Softplus().to(device)

    def forward(self, x, hidden, cell):

        x = copy.deepcopy(x)

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).to(self.device)


        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([batch_size, time_len, cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        
        ## layer normalization and back
        raw_esti_data = x[:,:,0]
        mean = torch.mean(raw_esti_data, axis=1)
        std = torch.std(raw_esti_data, axis=1)
        
        unsqueeze_mean = mean.unsqueeze(1).repeat(1,raw_esti_data.shape[1])
        unsqueeze_std = std.unsqueeze(1).repeat(1,raw_esti_data.shape[1])
        
        if self.normalize: x[:,:,0] =  (x[:,:,0]-unsqueeze_mean)/unsqueeze_std

        lstm_input = x
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # use h from all layers to calculate mu and sigma

        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)

        if self.output_scheme == "pred":

            pred = torch.squeeze(self.pred(hidden_permute))
            if self.normalize: pred += mean

            output = pred.view(-1,1)
        
        elif self.output_scheme == "mu_sigma":

            pre_sigma = self.distribution_presigma(hidden_permute)
            mu = torch.squeeze(self.distribution_mu(hidden_permute))
            sigma = torch.squeeze(self.distribution_sigma(pre_sigma))
            if self.normalize: 
                mu += mean
                sigma *= std

    
            output = torch.stack([mu, sigma], axis = 1)


        return output, hidden, cell
        

    def init_hidden(self, input_size):
        return torch.randn(self.LSTM_LAYER, input_size, self.LSTM_HIDDEN_DIM, device=self.device)

    def init_cell(self, input_size):
        return torch.randn(self.LSTM_LAYER, input_size, self.LSTM_HIDDEN_DIM, device=self.device)

    
    def predict(self, X_val):

        hidden = self.init_hidden(len(X_val))
        
        cell = self.init_cell(len(X_val))
        val_output = self(X_val, hidden, cell)[0]

        return val_output



class deep_ensemble_TS:

    def __init__(self,
                COV_DIM,
                LSTM_HIDDEN_DIM,
                layer = 1,
                n_models = 5,
                device = torch.device("cuda")
                 
    ):
        self.ensembles = []
        self.n_models = n_models
        self.device = device

        for i in range(n_models):

            self.ensembles.append(Vanilla_LSTM(
                COV_DIM = COV_DIM,
                LSTM_HIDDEN_DIM = LSTM_HIDDEN_DIM,
                device = device,
                LSTM_LAYER= layer,
                )    
            )


    def predict(
        self,
        X_val  
    ):
        if isinstance(X_val, np.ndarray):
            X_val = torch.Tensor(X_val).to(self.device)
            
        mu_sum = 0
        sigma_square_sum = 0
        mu_sigma_square_sum = 0

        for i in range(self.n_models):

            hidden = self.ensembles[i].init_hidden(len(X_val))
        
            cell = self.ensembles[i].init_cell(len(X_val))

            mu_sum += self.ensembles[i](X_val, hidden, cell)[0][:,:1]

            sigma_square_sum += self.ensembles[i](X_val, hidden, cell)[0][:,1:]**2

            mu_sigma_square_sum += self.ensembles[i](X_val, hidden, cell)[0][:,:1]**2 + self.ensembles[i](X_val, hidden, cell)[0][:,1:]**2

        
        mu_esti = mu_sum / self.n_models
        sigma_esti = torch.sqrt(mu_sigma_square_sum / self.n_models - mu_esti**2)
        # sigma_esti = torch.sqrt(sigma_square_sum / self.n_models)

        val_output = torch.cat((mu_esti, sigma_esti), axis = 1)

        return val_output

    