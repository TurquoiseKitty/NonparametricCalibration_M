from experiments.Conformal_Utils.OQR_helper import SYN_DATA, REAL_DATA
import experiments.Conformal_Utils.OQR_helper as helper
from argparse import Namespace
from experiments.Conformal_Utils.OQR_dataFetching import get_scaled_data
from NRC.losses import *
from NRC.models.DNNModels import MC_dropnet, Deep_Ensemble, vanilla_predNet
from NRC.train.DNNTrain import DNN_Trainer
import torch
import numpy as np
import pandas as pd
from experiments.Conformal_Utils import OQR_helper as helper
from experiments.Conformal_Utils.OQR_losses import batch_interval_loss
from experiments.Conformal_Utils.OQR_q_model_ens import QModelEns
from experiments.Conformal_Utils.OQRTrain import OQR_Trainer
import time
import copy
from NRC.utils import seed_all, iso_recal
from experiments.data_utils import common_processor_UCI


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
    "sharpness_90_muSigma": sharpness_90_muSigma,
    "sharpness_90": sharpness_90,
    "CheckScore": avg_pinball_quantile,
    "CheckScore_muSigma": avg_pinball_muSigma,
    "coverage_90_muSigma": coverage_90_muSigma,
    "coverage_90": coverage_90,
}

model_NameDict = {
    "HNN": MC_dropnet,
    "MCDrop": MC_dropnet,
    "DeepEnsemble": Deep_Ensemble,
    "ISR": MC_dropnet,
    "CQR": QModelEns,
    "OQR": QModelEns,
}


common_config = {

    "model_init": None,

    "model_config": {
        "n_input": 0,
        "device": torch.device('cuda'),
        "hidden_layers": [64, 64],
        "n_output": 0
    },

    "training_config": {
        "save_path_and_name": None,
        "LR": 1e-3,
        "Decay": 1e-6,
        "N_Epoch": 5000,
        "backdoor": None,
        "bat_size": 64,
        "early_stopping": True,
        "monitor_name": None,
        "patience": 100,
        "train_loss": None,
        "val_loss_criterias" : None,
        "monitor_name" : None,

        "validate_times": 50,
        "verbose": False,
    }
}
    
    
    
# the customizer will take in model name, and then customize everything
def config_customizer(modelname, n_input):

    configs = copy.deepcopy(common_config)
    configs["model_config"]["n_input"] = n_input

    if modelname == "HNN":

        configs["model_init"] = "HNN"
        configs["model_config"]["n_output"] = 2
        configs["model_config"]["drop_rate"] = 0.

        base_model = model_NameDict["HNN"](**configs["model_config"])

        configs["training_config"]["train_loss"] = mean_std_norm_loss
        configs["training_config"]["val_loss_criterias"] = {
            "nll" : mean_std_norm_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "nll"

    elif modelname == "MCDrop":
        
        configs["model_init"] = "MCDrop"
        configs["model_config"]["n_output"] = 2
        configs["model_config"]["drop_rate"] = 0.2

        base_model = model_NameDict["MCDrop"](**configs["model_config"])

        configs["training_config"]["train_loss"] = mean_std_norm_loss
        configs["training_config"]["val_loss_criterias"] = {
            "nll" : mean_std_norm_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "nll"
    
    elif modelname == "DeepEnsemble":
        
        configs["model_init"] = "DeepEnsemble"
        configs["model_config"]["n_output"] = 2
        configs["model_config"]["n_models"] = 5
    
        base_model = model_NameDict["DeepEnsemble"](**configs["model_config"])

        configs["training_config"]["train_loss"] = mean_std_forEnsemble
        configs["training_config"]["val_loss_criterias"] = {
            "nll" : mean_std_forEnsemble,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "nll"


    elif modelname == "ISR":

        # the base model is nothing other than an HNN
        configs["model_init"] = "ISR"
        configs["model_config"]["n_output"] = 2
        configs["model_config"]["drop_rate"] = 0.

        base_model = model_NameDict["ISR"](**configs["model_config"])

        configs["training_config"]["train_loss"] = mean_std_norm_loss
        configs["training_config"]["val_loss_criterias"] = {
            "nll" : mean_std_norm_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "nll"


    elif modelname in ["CQR", "OQR"]:

        backup_common = copy.deepcopy(common_config)

        configs = {}

        configs["model_config"] = {}
        configs["training_config"] = {}

        configs["model_config"]["input_size"] = n_input + 1
        configs["model_config"]["output_size"] = 1
        configs["model_config"]["hidden_size"] = backup_common["model_config"]["hidden_layers"][0]
        configs["model_config"]["num_layers"] = len(backup_common["model_config"]["hidden_layers"])
        configs["model_config"]["dropout"] = 0
        configs["model_config"]["lr"] = backup_common["training_config"]["LR"]
        configs["model_config"]["wd"] = backup_common["training_config"]["Decay"]
        configs["model_config"]["num_ens"] = 1
        configs["model_config"]["device"] = backup_common["model_config"]["device"]


        base_model = model_NameDict[modelname](**configs["model_config"])


        configs["training_config"] = backup_common["training_config"]

        if modelname == "CQR":

            configs["training_config"]["arg_corr_mult"] = 0

        elif modelname == "OQR":

            configs["training_config"]["arg_corr_mult"] = 0.5


        configs["training_config"]["train_loss"] = batch_interval_loss
        configs["training_config"]["val_loss_criterias"] = {
                'batch_int' : batch_interval_loss,
            }
        configs["training_config"]["monitor_name"] = "batch_int"


        
    else:

        raise NotImplementedError
    

    return base_model, configs

  

def testPerform_customizer_light(test_X, test_Y, model_name, model, \
                            aux_info):
    
    ret = {}
    
    if model_name in ["HNN", "MCDrop", "DeepEnsemble"]:

        val_criterias = ["MACE_muSigma", "AGCE_muSigma", "CheckScore_muSigma", "sharpness_90_muSigma", "coverage_90_muSigma"]       

        y_out = model.predict(test_X)

        for key in val_criterias:

            real_loss = loss_NameDict[key]

            real_err = real_loss(y_out, test_Y, q_list = np.linspace(0.05,0.95,7), light=True)

            if isinstance(real_err, torch.Tensor):

                real_err = real_err.item()

            ret[key] = real_err

    elif model_name in ["ISR"]:

        val_criterias = ["MACE_muSigma", "AGCE_muSigma", "CheckScore_muSigma", "sharpness_90_muSigma", "coverage_90_muSigma"]       

        y_out = model.predict(test_X)

        for key in val_criterias:

            real_loss = loss_NameDict[key]
            
            real_err = real_loss(y_out, test_Y, recal = True, recal_model = aux_info["recal_model"], q_list = np.linspace(0.05,0.95,7), light=True)

            if isinstance(real_err, torch.Tensor):

                real_err = real_err.item()

            ret[key] = real_err


    elif model_name in ["CQR", "OQR"]:

        quantiles = torch.Tensor(np.linspace(0.01,0.99,7))
        test_preds = model.predict_q(
            test_X, quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None
        )

        test_preds = torch.permute(test_preds, (1,0))
        
        val_criterias = ["MACE_Loss", "AGCE_Loss", "CheckScore", "sharpness_90", "coverage_90"]

        for key in val_criterias:

            real_loss = loss_NameDict[key]

            real_err = real_loss(test_preds, test_Y, q_list = np.linspace(0.01,0.99,7), light = True).item()

            if isinstance(real_err, torch.Tensor):

                real_err = real_err.item()

            ret[key] = real_err


    return ret



def run_benchmark(test_run = False):

    # ---------------------------configs for testing purpose---------------------------------#
    if test_run:

    # common_config["training_config"]["verbose"] = True
        common_config["training_config"]["N_Epoch"] = 1
        common_config["training_config"]["validate_times"] = 1
    # ---------------------------configs for testing purpose---------------------------------#

    big_df = {}


    seed_list = np.arange(5)

    for k in range(5):
                
        seed = seed_list[k]
        seed_all(1234 + seed)

        for dataname in  ['meps_19', 'meps_20', 'meps_21']:

            if dataname not in big_df.keys():

                big_df[dataname] = {}

            x, y = get_scaled_data(dataname)


            (train_X_raw, train_Y_raw), (recal_X_raw, recal_Y_raw), (val_X, val_Y), (test_X, test_Y) = common_processor_UCI(x, y, normal= False)

            for modelname in ["HNN", "MCDrop", "DeepEnsemble", "ISR", "CQR", "OQR"]:

                # train base model
                print("model: "+ modelname +" on data: "+dataname)

                star = time.time()

                aux_info = {
                            "recal_X": recal_X_raw,
                            "recal_Y": recal_Y_raw,
                        }

                # train the underlying

                train_X = train_X_raw
                train_Y = train_Y_raw

                base_model, configs = config_customizer(modelname, n_input = x.shape[1])
            
                train_X = torch.Tensor(train_X).to(configs["model_config"]["device"])
                val_X = torch.Tensor(val_X).to(configs["model_config"]["device"])
                test_X = torch.Tensor(test_X).to(configs["model_config"]["device"])
                train_Y = torch.Tensor(train_Y).to(configs["model_config"]["device"])
                val_Y = torch.Tensor(val_Y).to(configs["model_config"]["device"])
                test_Y = torch.Tensor(test_Y).to(configs["model_config"]["device"])


                if modelname in ["HNN", "MCDrop", "DeepEnsemble", "ISR"]:

                    DNN_Trainer(base_model, train_X, train_Y, val_X, val_Y, **configs["training_config"])

                
                elif modelname in ["CQR", "OQR"]:

                    OQR_Trainer(base_model, train_X, train_Y, val_X, val_Y, **configs["training_config"])

                else:

                    raise NotImplementedError
                

                ## some preparation operations
                if modelname == "ISR":

                    ISR_exp = np.linspace(0.01, 0.99, 100)

                    ISR_train_out = base_model(train_X)

                    ISR_pred_mat = mu_sig_toQuants(ISR_train_out[:,0],ISR_train_out[:,1],ISR_exp)

                    ISR_exp_quants, ISR_obs_quants = obs_vs_exp(y_true = train_Y, exp_quants = ISR_exp, quantile_preds = ISR_pred_mat)

                    ISR_recalibrator = iso_recal(ISR_exp_quants.cpu().numpy(), ISR_obs_quants.cpu().numpy())

                    aux_info["recal_model"] = ISR_recalibrator

                record = testPerform_customizer_light(test_X, test_Y, model_name= modelname, model = base_model, \
                                                aux_info = aux_info)

                second_layer_id = modelname

                for key in record.keys():

                    if second_layer_id+key not in big_df[dataname].keys():

                        big_df[dataname][second_layer_id+key] = []

                    big_df[dataname][second_layer_id+key].append(record[key])


                end = time.time()

                print("time spent:", end - star)


    mu_sig_df = {}
    for dataname in big_df.keys():

        if len(mu_sig_df) == 0:

            mu_sig_df['idxes'] = list(big_df[dataname].keys())

        mu_hold_list = []
        sig_hold_list = []

        for crit in big_df[dataname].keys():

            mu_hold_list.append(np.mean(big_df[dataname][crit]))
            sig_hold_list.append(np.std(big_df[dataname][crit]))


        mu_sig_df[dataname +"_mu"] = mu_hold_list
        mu_sig_df[dataname + "_std"] = sig_hold_list
        


    df = pd.DataFrame.from_dict(mu_sig_df)  

    df.to_csv("experiments/EXP2/record_bin/EXP2_benchmarks.csv",index=False)


if __name__ == "__main__":

    run_benchmark(False)

