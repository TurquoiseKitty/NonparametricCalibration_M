# code from https://docs.gpytorch.ai/en/latest/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html

import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from torch.utils.data import TensorDataset, DataLoader



class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class oneLayer_DeepGP(DeepGP):

    

    def __init__(self, n_input, hidden_layers, device = torch.device('cuda'), **kwargs):

        in_dim = n_input
        hidden_dim = hidden_layers[0]

        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=in_dim,
            output_dims=hidden_dim,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

        self.device = device

        self.to(device)

    def forward(self, inputs):

        assert isinstance(inputs, torch.Tensor)
        assert len(inputs.shape) == 2

        inputs = inputs.to(self.device)

        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, inputs, bat_size = 128):

        assert isinstance(inputs, torch.Tensor)
        assert len(inputs.shape) == 2

        val_set = TensorDataset(inputs)
        val_loader = DataLoader(val_set, batch_size=bat_size, shuffle=False)

        with torch.no_grad():
            mus = []
            variances = []
            
            for x_batch in val_loader:

                x_batch = x_batch[0].to(self.device)


                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)


        return torch.stack((torch.cat(mus, dim=-1).mean(dim = 0), torch.sqrt(torch.cat(variances, dim=-1).mean(dim = 0))), dim = 1)
