from NRC.utils import seed_all, iso_recal
from experiments.data_utils import get_uci_data, common_processor_UCI
from NRC.losses import *
from NRC.models.DNNModels import MC_dropnet, Deep_Ensemble, vanilla_predNet
from NRC.models.GPModels import oneLayer_DeepGP
from NRC.train.DNNTrain import DNN_Trainer
from NRC.train.GPTrain import DeepGP_Trainer
from NRC.kernel_methods import tau_to_quant_datasetCreate
import copy
import torch
import numpy as np
import pandas as pd
from experiments.Conformal_Utils import OQR_helper as helper
from experiments.Conformal_Utils.OQR_losses import batch_interval_loss
from experiments.Conformal_Utils.OQR_q_model_ens import QModelEns
from experiments.Conformal_Utils.OQRTrain import OQR_Trainer
import time

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
    "CCL": MC_dropnet,
    "ISR": MC_dropnet,
    "DGP": oneLayer_DeepGP,
    "MMD": MC_dropnet,
    "MAQR": vanilla_predNet,
    "CQR": QModelEns,
    "OQR": QModelEns,
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
        "Decay": 1e-6,
        "N_Epoch": 500,
        "backdoor": None,
        "bat_size": 16,
        "early_stopping": True,
        "monitor_name": None,
        "patience": 20,
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
    
    elif modelname ==  "CCL":
        
        configs["model_init"] = "CCL"
        configs["model_config"]["n_output"] = 2
        configs["model_config"]["drop_rate"] = 0.
    
        base_model = model_NameDict["CCL"](**configs["model_config"])

        configs["training_config"]["train_loss"] = BeyondPinball_muSigma
        configs["training_config"]["val_loss_criterias"] = {
            "beyondPinBall" : BeyondPinball_muSigma,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "beyondPinBall"

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

    elif modelname == "DGP":
        
        configs["model_init"] = "DGP"
        configs["hidden_layers"] = [10]

        base_model = model_NameDict["DGP"](**configs["model_config"])

        configs["training_config"]["LR"] = 1e-2
        configs["training_config"]["N_Epoch"] = 100
        configs["training_config"]["patience"] = 10
        configs["training_config"]["train_loss"] = mean_std_norm_loss
        configs["training_config"]["val_loss_criterias"] = {
            "nll" : mean_std_norm_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "nll"
        configs["training_config"]["num_samples"] = 5
    
    elif modelname ==  "MMD":

        configs["model_init"] = "MMD"
        configs["model_config"]["n_output"] = 2
        configs["model_config"]["drop_rate"] = 0.

        base_model = model_NameDict["MMD"](**configs["model_config"])

        configs["training_config"]["train_loss"] = mean_std_norm_loss
        configs["training_config"]["val_loss_criterias"] = {
            "nll" : mean_std_norm_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "nll"

    elif modelname == "MMD_aux":

        configs["model_init"] = "MMD"
        configs["model_config"]["n_output"] = 2
        configs["model_config"]["drop_rate"] = 0.

        base_model = None

        configs["training_config"]["train_loss"] = MMD_Loss
        configs["training_config"]["val_loss_criterias"] = {
            "MMD": MMD_Loss,
            "nll": mean_std_norm_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "MMD"

    elif modelname == "MAQR":

        configs["model_init"] = "MAQR"
        configs["model_config"]["n_output"] = 1

        base_model = model_NameDict["MAQR"](**configs["model_config"])

        configs["training_config"]["train_loss"] = mse_loss
        configs["training_config"]["val_loss_criterias"] = {
            "mse": mse_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "mse"
    
    elif modelname == "MAQR_aux":

        configs["model_init"] = "MAQR"
        configs["model_config"]["n_output"] = 1

        base_model = None
        configs["training_config"]["bat_size"] = 1024

        configs["training_config"]["train_loss"] = mse_loss
        configs["training_config"]["val_loss_criterias"] = {
            "mse": mse_loss,
            "rmse": rmse_loss
        }
        configs["training_config"]["monitor_name"] = "mse"

        configs["wid"] = 10

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

  

def testPerform_customizer(test_X, test_Y, model_name, model, \
                            aux_info):
    
    ret = {}
    
    if model_name in ["HNN", "MCDrop", "DeepEnsemble", "CCL", "DGP", "MMD"]:

        val_criterias = ["MACE_muSigma", "AGCE_muSigma", "CheckScore_muSigma", "sharpness_90_muSigma", "coverage_90_muSigma"]       

        y_out = model.predict(test_X)

        for key in val_criterias:

            real_loss = loss_NameDict[key]

            real_err = real_loss(y_out, test_Y)

            if isinstance(real_err, torch.Tensor):

                real_err = real_err.item()

            ret[key] = real_err

    elif model_name in ["ISR"]:

        val_criterias = ["MACE_muSigma", "AGCE_muSigma", "CheckScore_muSigma", "sharpness_90_muSigma", "coverage_90_muSigma"]       

        y_out = model.predict(test_X)

        for key in val_criterias:

            real_loss = loss_NameDict[key]
            
            real_err = real_loss(y_out, test_Y, recal = True, recal_model = aux_info["recal_model"])

            if isinstance(real_err, torch.Tensor):

                real_err = real_err.item()

            ret[key] = real_err

    elif model_name in ["MAQR"]:

        model_quantpred = aux_info["model_quantpred"]

        quants = np.linspace(0.01, 0.99, 100)

        MAQR_quant_bed = torch.Tensor(quants).view(-1, 1).repeat(1, len(test_X)).view(-1).to(test_X.device)

        MAQR_test_X_stacked = test_X.repeat(len(quants), 1)

        MAQR_forReg_X = torch.cat([MAQR_test_X_stacked, MAQR_quant_bed.view(-1,1)], dim=1)

        MAQR_pred_eps = model_quantpred(MAQR_forReg_X)

        MAQR_pred_eps = MAQR_pred_eps.reshape(len(quants), len(test_Y))

        MAQR_test_mean = model(test_X)

        pred_Y = MAQR_pred_eps + MAQR_test_mean.view(1,-1).repeat(len(MAQR_pred_eps),1)

        val_criterias = ["MACE_Loss", "AGCE_Loss", "CheckScore", "sharpness_90", "coverage_90"]

        for key in val_criterias:

            real_loss = loss_NameDict[key]

            real_err = real_loss(pred_Y, test_Y, q_list = np.linspace(0.01,0.99,100)).item()

            if isinstance(real_err, torch.Tensor):

                real_err = real_err.item()

            ret[key] = real_err

    elif model_name in ["CQR", "OQR"]:

        quantiles = torch.Tensor(np.linspace(0.01,0.99,100))
        test_preds = model.predict_q(
            test_X, quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None
        )

        test_preds = torch.permute(test_preds, (1,0))
        
        val_criterias = ["MACE_Loss", "AGCE_Loss", "CheckScore", "sharpness_90", "coverage_90"]

        for key in val_criterias:

            real_loss = loss_NameDict[key]

            real_err = real_loss(test_preds, test_Y, q_list = np.linspace(0.01,0.99,100)).item()

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

    dataset_path = "datasets/UCI_datasets"

    seed_list = np.arange(5)

    for k in range(5):
                
        seed = seed_list[k]
        seed_all(1234 + seed)

        for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:
        # for dataname in ["boston"]:

            if dataname not in big_df.keys():

                big_df[dataname] = {}

            x, y = get_uci_data(data_name= dataname, dir_name= dataset_path)

            (train_X_raw, train_Y_raw), (recal_X_raw, recal_Y_raw), (val_X, val_Y), (test_X, test_Y) = common_processor_UCI(x, y,shrink=True)

            for modelname in ["HNN", "MCDrop", "DeepEnsemble", "CCL", "ISR", "DGP", "MMD", "MAQR", "CQR", "OQR"]:

                # train base model
                print("model: "+ modelname +" on data: "+dataname)

                star = time.time()

                aux_info = {
                            "recal_X": recal_X_raw,
                            "recal_Y": recal_Y_raw,
                        }

                # train the underlying

                if not modelname in ["MAQR", "NRC", "NRC-RF", "NRC-RP", "NRC-Cov"]:
                    # for benchmarks, we combine the training set and the recalibration set

                    train_X = np.concatenate((train_X_raw, recal_X_raw), axis = 0)
                    train_Y = np.concatenate((train_Y_raw, recal_Y_raw), axis = 0)

                else:

                    train_X = train_X_raw
                    train_Y = train_Y_raw

                base_model, configs = config_customizer(modelname, n_input = x.shape[1])
            
                train_X = torch.Tensor(train_X).to(configs["model_config"]["device"])
                val_X = torch.Tensor(val_X).to(configs["model_config"]["device"])
                test_X = torch.Tensor(test_X).to(configs["model_config"]["device"])
                train_Y = torch.Tensor(train_Y).to(configs["model_config"]["device"])
                val_Y = torch.Tensor(val_Y).to(configs["model_config"]["device"])
                test_Y = torch.Tensor(test_Y).to(configs["model_config"]["device"])


                if modelname in ["HNN", "MCDrop", "DeepEnsemble", "CCL", "ISR", "MMD", "MAQR"]:

                    DNN_Trainer(base_model, train_X, train_Y, val_X, val_Y, **configs["training_config"])

                elif modelname in ["DGP"]:

                    DeepGP_Trainer(base_model, train_X, train_Y, val_X, val_Y, **configs["training_config"])

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


                elif modelname == "MMD":

                    model_reuse = copy.deepcopy(base_model)

                    _, aux_config =  config_customizer("MMD_aux", n_input = x.shape[1])

                    DNN_Trainer(model_reuse, train_X, train_Y, val_X, val_Y, **aux_config["training_config"])

                    base_model = copy.deepcopy(model_reuse)

                elif modelname == "MAQR":

                    _, aux_config =  config_customizer("MAQR_aux", n_input = x.shape[1] + 1)

                    wid = aux_config["wid"]

                    MAQR_exp = np.linspace(0.01, 0.99, 100)

                    recal_X = torch.Tensor(recal_X_raw).to(configs["model_config"]["device"])
                    recal_Y = torch.Tensor(recal_Y_raw).to(configs["model_config"]["device"])

                    MAQR_eps = (recal_Y - base_model(recal_X).view(-1)).detach().cuda()


                    MAQR_reg_X, MAQR_reg_Y = tau_to_quant_datasetCreate(recal_X, epsilon=MAQR_eps, quants= np.linspace(0.01,0.99,20),wid = wid)

                    model_quantpred = model_NameDict["MAQR"](**aux_config["model_config"])

                    train_amount = int(len(MAQR_reg_Y) * 9 / 10)

                    
                    DNN_Trainer(model_quantpred, MAQR_reg_X[:train_amount], MAQR_reg_Y[:train_amount], MAQR_reg_X[train_amount:], MAQR_reg_Y[train_amount:], **aux_config["training_config"])

                    aux_info["model_quantpred"] = model_quantpred



                record = testPerform_customizer(test_X, test_Y, model_name= modelname, model = base_model, \
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

    df.to_csv("experiments/EXP1/record_bin/EXP1_benchmarks.csv",index=False)


if __name__ == "__main__":

    run_benchmark(False)







                









