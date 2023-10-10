# refer to https://github.com/JavierAntoran/Bayesian-Neural-Networks

# MC dropout for heteroskedastic network.


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader



class raw_net(nn.Module):

    def __init__(
            self,
            n_input,
            hidden_layers,
            drop_rate = 0.5,
            device = torch.device('cuda'),
            **kwargs
    ):
        super(raw_net, self).__init__()

        self.n_input = n_input
        self.hidden_layers = hidden_layers
        self.device = device
        self.drop_rate = drop_rate

        model_seq = []

        prev_dim = n_input
        for dimi in hidden_layers:

            model_seq.append(nn.Linear(prev_dim, dimi))
            model_seq.append(nn.LeakyReLU(0.2, inplace=True))

            if self.drop_rate > 1E-4:
                model_seq.append(nn.Dropout(drop_rate))

            prev_dim = dimi

        self.raw_model = nn.ModuleList(model_seq).to(self.device)


    def forward(
            self, x, **kwargs
    ):
        raise NotImplementedError
    

    def predict(
            self, x, **kwargs
    ):
        raise NotImplementedError
    

class vanilla_predNet(raw_net):

    def __init__(
            self,
            n_input,
            hidden_layers,
            n_output = 1,
            device = torch.device('cuda'),
            **kwargs
    ): 

        super(vanilla_predNet, self).__init__(
            n_input= n_input,
            hidden_layers= hidden_layers,
            drop_rate= 0,
            device= device
        )

        self.n_output = n_output

        if len(hidden_layers) > 0:

            dim_before = hidden_layers[-1]
        else:
            dim_before = n_input

        self.tail = nn.Linear(dim_before, n_output).to(self.device)


    def forward(self, x: torch.Tensor):

        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        x = x.to(self.device)

        for m in self.raw_model:

            x = m(x)

        x = self.tail(x)

        return x
    

    def predict(self, 
                x: torch.Tensor,
                bat_size = 128,
                ):
        
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        # sometimes validate set might get too big

        val_set = TensorDataset(x)
        val_loader = DataLoader(val_set, batch_size=bat_size, shuffle=False)

        with torch.no_grad():
            mus = []

            for x_batch in val_loader:


                x = x_batch[0].to(self.device)

                out = self(x)                

                mus.append(out[:,0])


        return torch.cat(mus, dim=-1)
    
    def feature_layer(self, 
                x: np.array):
        
        x = torch.Tensor(x).to(self.device)

        for m in self.raw_model:

            x = m(x)

        return x.detach().cpu().numpy()
        





class quantile_predNet(vanilla_predNet):

    
    def __init__(
            self,
            n_input,
            hidden_layers,
            n_output,
            device = torch.device('cuda'),
            **kwargs
    ): 

        super(quantile_predNet, self).__init__(
            n_input= n_input,
            hidden_layers= hidden_layers,
            n_output= n_output,
            device= device
        )


    def predict(self, 
                x: torch.Tensor,
                bat_size = 128,
                ):
        
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        # sometimes validate set might get too big

        val_set = TensorDataset(x)
        val_loader = DataLoader(val_set, batch_size=bat_size, shuffle=False)

        with torch.no_grad():
            outs = []

            for x_batch in val_loader:


                x = x_batch[0].to(self.device)

                out = self(x)                

                outs.append(out)


        return torch.cat(outs, dim=0)







class MC_dropnet(raw_net):

    def __init__(
            self,
            n_input,
            hidden_layers,
            n_output = 2,
            drop_rate = 0.1,
            device = torch.device('cuda'),
            **kwargs
    ): 
        
        # we only implement heteroskedastic setting
        assert n_output == 2

        super(MC_dropnet, self).__init__(
            n_input= n_input,
            hidden_layers= hidden_layers,
            drop_rate= drop_rate,
            device= device
        )

        self.n_output = n_output

        if len(hidden_layers) > 0:

            dim_before = hidden_layers[-1]
        else:
            dim_before = n_input

        self.tail = nn.Linear(dim_before, n_output).to(self.device)


    def forward(self, x: torch.Tensor):

        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        x = x.to(self.device)

        for m in self.raw_model:

            x = m(x)

        x = self.tail(x)

        mu = x[:, :1]
        raw_sigma = x[:, 1:]
        sigma = nn.Softplus()(raw_sigma)


        return torch.cat((mu, sigma), axis = 1)
    

    def predict(self, 
                x: torch.Tensor,
                bat_size = 128,
                trial = 100):
        
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        # sometimes validate set might get too big

        val_set = TensorDataset(x)
        val_loader = DataLoader(val_set, batch_size=bat_size, shuffle=False)

        with torch.no_grad():
            mus = []
            sigs = []

            for x_batch in val_loader:


                x = x_batch[0].to(self.device)

                if self.drop_rate > 1E-4:
                    samples = []
                    noises = []

                    for i in range(trial):
                        preds = self(x)
                        samples.append(preds[:, 0])
                        noises.append(preds[:, 1])

                    samples = torch.stack(samples, dim = 0)
                    noises = torch.stack(noises, dim = 0)
                    assert samples.shape == (trial, len(x))

                    mean_preds = samples.mean(dim = 0)
                    aleatoric = (noises**2).mean(dim = 0)**0.5

                    epistemic = samples.var(dim = 0)**0.5
                    total_unc = (aleatoric**2 + epistemic**2)**0.5

                    out = torch.stack((mean_preds, aleatoric), dim = 1)



                else:
                    out = self(x)

                

                mus.append(out[:,0])
                sigs.append(out[:,1])

        return  torch.stack((torch.cat(mus, dim=-1), torch.cat(sigs, dim=-1)), dim = 1)


class Deep_Ensemble(raw_net):

    def __init__(
            self,
            n_input,
            hidden_layers,
            n_output = 2,
            n_models = 5,
            device = torch.device('cuda'),
            **kwargs
    ):
        
        assert n_output == 2


        super(Deep_Ensemble, self).__init__(
            n_input= 2,
            hidden_layers= hidden_layers,
            drop_rate= 0,
            device= device
        )
        
        ensembles_list = []
        self.n_models = n_models
        self.device = device

        for i in range(n_models):

            ensembles_list.append(
                MC_dropnet(
                n_input= n_input,
                hidden_layers= hidden_layers,
                n_output= n_output,
                drop_rate= 0.,
                device= device
                )
            )
        
        self.ensembles = nn.ModuleList(ensembles_list)

    def forward(self, x: torch.Tensor):

        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        x = x.to(self.device)

        fin_output = []

        for mc_model in self.ensembles:

            out = mc_model(x)

            fin_output.append(out)

        
        return torch.cat(fin_output, dim= 0)


    def predict(self, x: torch.Tensor):
        
        raw_out = self(x)


        assert raw_out.shape == (self.n_models * len(x), 2)

        splitted = torch.stack(torch.split(raw_out, len(x)), dim = 0)

        samples = splitted[:, :, 0]
        noises = splitted[:, :, 1]

        mean_preds = samples.mean(dim = 0)
        aleatoric = (noises**2).mean(dim = 0)**0.5

        return torch.stack((mean_preds, aleatoric), dim = 1)






      

    
        