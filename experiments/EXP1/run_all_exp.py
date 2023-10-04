from NRC.utils import seed_all
from experiments.data_utils import get_uci_data, common_processor_UCI
from NRC.losses import mse_loss, rmse_loss, mean_std_norm_loss, mean_std_forEnsemble, BeyondPinball_muSigma, MMD_Loss, MACE_Loss, MACE_muSigma, AGCE_Loss, AGCE_muSigma, avg_pinball_quantile, avg_pinball_muSigma

from NRC.models.DNNModels import MC_dropnet, Deep_Ensemble, vanilla_predNet
from NRC.models.GPModels import oneLayer_DeepGP
from NRC.train.DNNTrain import DNN_Trainer
import copy
import torch
import numpy as np


loss_NameDict = {
    "mse_loss": mse_loss, 
    "rmse_loss": rmse_loss, 
    "mean_std_norm_loss": mean_std_norm_loss, 
    "mean_std_forEnsemble": mean_std_forEnsemble, 
    "BeyondPinball_muSigma": BeyondPinball_muSigma,
    "MMD_Loss": MMD_Loss,
    "MACE_Loss": MACE_Loss,
    "MACE_muSigma": MACE_muSigma,
    "AGCE_Loss": AGCE_Loss,
    "AGCE_muSigma": AGCE_muSigma,
    "CheckScore": avg_pinball_quantile,
    "CheckScore_muSigma": avg_pinball_muSigma
}

model_NameDict = {
    "HNN": MC_dropnet,
    "MCDrop": MC_dropnet,
    "DeepEnsemble": Deep_Ensemble,
    "CCL": MC_dropnet,
    "ISR": MC_dropnet,
    "DGP": oneLayer_DeepGP,
    "MMD": MC_dropnet,
    "MAQR": vanilla_predNet,
    "CQR": None,
    "OQR": None,
    "NRC": None, 
    "NRC-RF": None, 
    "NRC-RP": None, 
    "NRC-Cov": None
}


common_config = {

    "model_init": None,

    "model_config": {
        "n_input": 0,
        "device": torch.device('cuda'),
        "hidden_layers": [20, 20],
        "n_output": 0
    },

    "training_config": {
        "save_path_and_name": None,
        "LR": 5e-3,
        "Decay": 1e-4,
        "N_Epoch": 2000,
        "backdoor": None,
        "bat_size": 16,
        "early_stopping": True,
        "monitor_name": None,
        "patience": 200,
        "train_loss": None,
        "val_loss_criterias" : None,
        "monitor_name" : None,

        "validate_times": 50,
        "verbose": True,
    }
}
    
    
    
# the customizer will take in model name, and then customize everything
def config_customizer(modelname, n_input):

    configs = copy.deepcopy(common_config)
    configs["model_config"]["n_input"] = n_input

    if modelname == "HNN":

        configs["model_init"] = "HNN"
        configs["model_config"]["n_output"] = 2
        configs["model_config"]["drop_rate"] = 0.1

        base_model = MC_dropnet(**configs["model_config"])

        configs["training_config"]["train_loss"] = mean_std_norm_loss
        configs["training_config"]["val_loss_criterias"] = {
            "nll" : mean_std_norm_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "nll"
        

    else:

        raise NotImplementedError
    

    return base_model, configs

  





if __name__ == "__main__":

    dataset_path = "datasets/UCI_datasets"

    # for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:
    for dataname in ["boston"]:

        x, y = get_uci_data(data_name= dataname, dir_name= dataset_path)

        (train_X, train_Y), (recal_X, recal_Y), (val_X, val_Y), (test_X, test_Y) = common_processor_UCI(x, y)

        # for modelname in ["HNN", "MCDrop", "DeepEnsemble", "CCL", "ISR", "DGP", "MMD", "MAQR", "CQR", "OQR", "NRC", "NRC-RF", "NRC-RP", "NRC-Cov"]:
        for modelname in ["HNN"]:


            # for seed in range(5):
            for seed in range(1):

                seed_all(seed)

                # train the underlying

                if not modelname in ["NRC", "NRC-RF", "NRC-RP", "NRC-Cov"]:
                    # for benchmarks, we combine the training set and the recalibration set

                    train_X = np.concatenate((train_X, recal_X), axis = 0)
                    train_Y = np.concatenate((train_Y, recal_Y), axis = 0)


                base_model, configs = config_customizer(modelname, n_input = x.shape[1])

                train_X = torch.Tensor(train_X).to(configs["model_config"]["device"])
                val_X = torch.Tensor(val_X).to(configs["model_config"]["device"])
                test_X = torch.Tensor(test_X).to(configs["model_config"]["device"])
                train_Y = torch.Tensor(train_Y).to(configs["model_config"]["device"])
                val_Y = torch.Tensor(val_Y).to(configs["model_config"]["device"])
                test_Y = torch.Tensor(test_Y).to(configs["model_config"]["device"])


                if not modelname in ["DGP"]:

                    DNN_Trainer(base_model, train_X, train_Y, val_X, val_Y, **configs["training_config"])






                









