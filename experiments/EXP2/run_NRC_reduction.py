from NRC.utils import seed_all, iso_recal
from experiments.data_utils import get_uci_data, common_processor_UCI
from experiments.Conformal_Utils.OQR_dataFetching import get_scaled_data
from experiments.Conformal_Utils.OQR_losses import batch_interval_loss
from NRC.losses import *
from NRC.models.DNNModels import MC_dropnet, Deep_Ensemble, vanilla_predNet
from NRC.train.DNNTrain import DNN_Trainer
from NRC.kernel_methods import tau_to_quant_datasetCreate, kernel_estimator
import copy
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import random_projection
import time
from sklearn.preprocessing import StandardScaler
from experiments.Conformal_Utils.OQRTrain import OQR_Trainer
from experiments.Conformal_Utils.OQR_q_model_ens import QModelEns


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

    "NRC": vanilla_predNet, 
    "NRC-RP": vanilla_predNet, 
    "NRC-RF-RP": None,
    "NRC-Layer": vanilla_predNet,
    "NRC-CQR-RP": QModelEns
}


common_config = {

    "model_init": None,

    "model_config": {
        "n_input": 0,
        "device": torch.device('cuda'),
        "hidden_layers": [64, 30],
        "n_output": 1
    },

    "training_config": {
        "save_path_and_name": None,
        "LR": 1e-3,
        "Decay": 1e-6,
        "N_Epoch": 5000,
        "backdoor": None,
        "bat_size": 1024,
        "early_stopping": True,
        "patience": 100,

        "train_loss": mse_loss,
        "val_loss_criterias":{
            "mse": mse_loss,
            "rmse": rmse_loss
        },
        "monitor_name": "mse",

        "validate_times": 50,
        "verbose": False,
    },

    "wid": 10
}
    
    
    
# the customizer will take in model name, and then customize everything
def NRC_config_customizer(modelname, n_input):

    configs = copy.deepcopy(common_config)
    configs["model_config"]["n_input"] = n_input
    configs["model_init"] = modelname

    base_model = None

    if modelname in ["NRC", "NRC-RP", "NRC-Layer"]:

        base_model = model_NameDict[modelname](**configs["model_config"])

    elif modelname in ["NRC-CQR-RP"]:

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

        configs["training_config"]["arg_corr_mult"] = 0

        configs["training_config"]["train_loss"] = batch_interval_loss
        configs["training_config"]["val_loss_criterias"] = {
                'batch_int' : batch_interval_loss,
            }
        configs["training_config"]["monitor_name"] = "batch_int"

    else:

        raise NotImplementedError
    
    return base_model, configs

  

def NRC_testPerform_customizer_light(test_X, test_Y, model_name, model, \
                            aux_info):
    
    reformer = aux_info['reformer']
    
    val_criterias = [
        "MACE_Loss", "AGCE_Loss", "CheckScore", "sharpness_90", "coverage_90"
    ]
    
    ret = {}

    wid = aux_info['wid']

    recal_X = copy.deepcopy(aux_info["recal_X"])
    recal_Y = copy.deepcopy(aux_info["recal_Y"])

    if model_name in ["NRC-RF-RP"]:

        recal_mean = torch.Tensor(model.predict(recal_X)).to(aux_info["device"])
        test_mean = torch.Tensor(model.predict(test_X.cpu().numpy())).to(aux_info["device"])

    elif model_name in ["NRC-RP", "NRC-Layer"]:

        recal_mean = model.predict(torch.Tensor(recal_X).to(aux_info["device"])).view(-1)
        test_mean = model.predict(test_X).view(-1)


    elif model_name in ["NRC-CQR-RP"]:

        recal_preds = torch.permute(model.predict_q(
            torch.Tensor(recal_X).to(aux_info["device"]), np.linspace(0.05,0.95,7), ens_pred_type='conf',
            recal_model=None, recal_type=None
        ), (1, 0))

        test_preds = torch.permute(model.predict_q(
            test_X, np.linspace(0.05,0.95,7), ens_pred_type='conf',
            recal_model=None, recal_type=None
        ), (1, 0))

    else:

        raise NotImplementedError
    

    if model_name in ["NRC-RP", "NRC-RF-RP", "NRC-CQR-RP"]:

        test_X =  reformer(test_X.cpu().numpy())

        recal_X = reformer(recal_X)


    if isinstance(test_X, torch.Tensor):

        test_X = test_X.cpu().numpy()


    if model_name in ["NRC-RF-RP", "NRC-RP", "NRC-Layer"]:

        eps_diffQuants = kernel_estimator(
            test_Z = test_X,
            recal_Z = recal_X,
            recal_epsilon = recal_Y - recal_mean.cpu().numpy(),
            quants = np.linspace(0.05,0.95,7),
            wid= wid
        )

        eps_diffQuants = torch.Tensor(eps_diffQuants).to(aux_info["device"])

        y_diffQuants = eps_diffQuants + test_mean.view(1,-1).repeat(len(eps_diffQuants),1)

    elif model_name in ["NRC-CQR-RP"]:

        light_quantiles = np.linspace(0.05,0.95,7)

        y_diffQuants = np.zeros((7, len(test_Y)))

        for i in range(7):
  
            eps_diffQuants = kernel_estimator(
                test_Z = test_X,
                recal_Z = recal_X,
                recal_epsilon = recal_Y - recal_preds[i].cpu().numpy(),
                quants = np.array([light_quantiles[i]]),
                wid= wid
            )

            y_diffQuants[i] = eps_diffQuants[0] + test_preds[i].cpu().numpy()

        y_diffQuants = torch.Tensor(y_diffQuants).to(aux_info["device"])

    else:

        raise NotImplementedError




    for key in val_criterias:

        real_loss = loss_NameDict[key]

        real_err = real_loss(y_diffQuants, test_Y, q_list = np.linspace(0.05,0.95,7), light=True).item()

        if isinstance(real_err, torch.Tensor):

            real_err = real_err.item()

        ret[key] = real_err




    return ret


def run_NRC(test_run = False):

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
        seed_all(seed)
    

        for dataname in  ['meps_19', 'meps_20', 'meps_21', 'facebook_1', 'facebook_2', 'blog_data']:
        # for dataname in  ['facebook_1', 'facebook_2']:

            if dataname not in big_df.keys():

                big_df[dataname] = {}

            x, y = get_scaled_data(dataname)


            (train_X_raw, train_Y_raw), (recal_X_raw, recal_Y_raw), (val_X, val_Y), (test_X, test_Y) = common_processor_UCI(x, y, normal= False)

            for modelname in ["NRC-RP", "NRC-RF-RP", "NRC-Layer", "NRC-CQR-RP"]:
            # for modelname in ["NRC-CQR-RP"]:

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

                base_model, configs = NRC_config_customizer(modelname, n_input = x.shape[1])

                aux_info['device'] = configs["model_config"]["device"]
            
                train_X = torch.Tensor(train_X).to(configs["model_config"]["device"])
                val_X = torch.Tensor(val_X).to(configs["model_config"]["device"])
                test_X = torch.Tensor(test_X).to(configs["model_config"]["device"])
                train_Y = torch.Tensor(train_Y).to(configs["model_config"]["device"])
                val_Y = torch.Tensor(val_Y).to(configs["model_config"]["device"])
                test_Y = torch.Tensor(test_Y).to(configs["model_config"]["device"])


                if modelname in ["NRC-RP", "NRC-Layer"]:

                    DNN_Trainer(base_model, train_X, train_Y, val_X, val_Y, **configs["training_config"])

                elif modelname in ["NRC-RF-RP"]:

                    depth = 10

                    base_model = RandomForestRegressor(max_depth=depth, random_state=0)
                    base_model.fit(train_X.cpu().numpy(), train_Y.cpu().numpy())

                elif modelname in ["NRC-CQR-RP"]:

                    OQR_Trainer(base_model, train_X, train_Y, val_X, val_Y, **configs["training_config"])

                else:

                    raise NotImplementedError


                # some additional operations

                n_component = 30
                

                if modelname in ["NRC-RP", "NRC-RF-RP", "NRC-CQR-RP"]:


                    transformer = random_projection.GaussianRandomProjection(n_components = n_component)

                    reformer_unnorm = lambda x : transformer.fit_transform(x)

                    # normalize

                    normalizer = StandardScaler().fit(reformer_unnorm(recal_X_raw))

                    reformer = lambda x : normalizer.fit_transform(transformer.fit_transform(x))

                else:

                    device = configs["model_config"]["device"]

                    reformer_unnorm = lambda x : base_model.feature_layer(recal_X_raw)

                    normalizer = StandardScaler().fit(reformer_unnorm(recal_X_raw))

                    reformer = lambda x : normalizer.fit_transform(reformer_unnorm(x))


   

                aux_info['reformer'] = reformer

                # many choices of wid

                for wid in [10, 30, 50, 100]:


                    aux_info['wid'] = wid

                    record = NRC_testPerform_customizer_light(test_X, test_Y, model_name= modelname, model = base_model, \
                                                aux_info = aux_info)


                    second_layer_id = modelname + "_w" + str(wid) +"_"

                    for key in record.keys():

                        if second_layer_id+key not in big_df[dataname].keys():

                            big_df[dataname][second_layer_id+key] = []

                        big_df[dataname][second_layer_id+key].append(record[key])


                end = time.time()

                print("time spent: ", end - star)



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

    df.to_csv("experiments/EXP2/record_bin/EXP2_NRCexperiments.csv",index=False)



if __name__ == "__main__":

    run_NRC(False)




                









