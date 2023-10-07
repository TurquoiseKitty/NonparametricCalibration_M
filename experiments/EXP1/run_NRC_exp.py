from NRC.utils import seed_all, iso_recal
from experiments.data_utils import get_uci_data, common_processor_UCI
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
    "NRC-RF": None, 
    "NRC-RP": vanilla_predNet, 
    "NRC-Cov":  vanilla_predNet,
    "PureKernel": None,
    "NRC-RF-RP": None,
}


common_config = {

    "model_init": None,

    "model_config": {
        "n_input": 0,
        "device": torch.device('cuda'),
        "hidden_layers": [20, 20],
        "n_output": 1
    },

    "training_config": {
        "save_path_and_name": None,
        "LR": 5e-3,
        "Decay": 1e-6,
        "N_Epoch": 2000,
        "backdoor": None,
        "bat_size": 16,
        "early_stopping": True,
        "patience": 200,

        "train_loss": mse_loss,
        "val_loss_criterias":{
            "mse": mse_loss,
            "rmse": rmse_loss
        },
        "monitor_name": "mse",

        "validate_times": 50,
        "verbose": False,
    },

    "wid": 5
}
    
    
    
# the customizer will take in model name, and then customize everything
def NRC_config_customizer(modelname, n_input):

    configs = copy.deepcopy(common_config)
    configs["model_config"]["n_input"] = n_input
    configs["model_init"] = modelname

    base_model = None

    if modelname in ["NRC", "NRC-RP", "NRC-Cov"]:

        base_model = model_NameDict[modelname](**configs["model_config"])
    

    return base_model, configs

  

def NRC_testPerform_customizer(test_X, test_Y, model_name, model, \
                            aux_info):
    
    reformer = aux_info['reformer']
    
    val_criterias = [
        "MACE_Loss", "AGCE_Loss", "CheckScore", "sharpness_90", "coverage_90"
    ]
    
    ret = {}

    wid = aux_info['wid']

    recal_X = copy.deepcopy(aux_info["recal_X"])
    recal_Y = copy.deepcopy(aux_info["recal_Y"])

    if model_name in ["NRC-RF", "NRC-RF-RP"]:

        recal_mean = torch.Tensor(model.predict(recal_X)).to(aux_info["device"])
        test_mean = torch.Tensor(model.predict(test_X.cpu().numpy())).to(aux_info["device"])

    elif model_name == "PureKernel":

        recal_mean = torch.zeros(len(recal_X)).to(aux_info["device"])
        test_mean = torch.zeros(len(test_X)).to(aux_info["device"])

    else:


        recal_mean = model.predict(torch.Tensor(recal_X).to(aux_info["device"])).view(-1)
        test_mean = model.predict(test_X).view(-1)

    if model_name in ["NRC-RP", "NRC-Cov", "NRC-RF-RP"]:

        test_X =  reformer(test_X.cpu().numpy())

        recal_X = reformer(recal_X)


    if isinstance(test_X, torch.Tensor):

        test_X = test_X.cpu().numpy()


    eps_diffQuants = kernel_estimator(
        test_Z = test_X,
        recal_Z = recal_X,
        recal_epsilon = recal_Y - recal_mean.cpu().numpy(),
        quants = np.linspace(0.01,0.99,100),
        wid= wid
    )

    eps_diffQuants = torch.Tensor(eps_diffQuants).to(aux_info["device"])

    y_diffQuants = eps_diffQuants + test_mean.view(1,-1).repeat(len(eps_diffQuants),1)

    for key in val_criterias:

        real_loss = loss_NameDict[key]

        real_err = real_loss(y_diffQuants, test_Y, q_list = np.linspace(0.01,0.99,100)).item()

        if isinstance(real_err, torch.Tensor):

            real_err = real_err.item()

        ret[key] = real_err




    return ret


def run_NRC():

    # ---------------------------configs for testing purpose---------------------------------#
    # common_config["training_config"]["verbose"] = True
    # common_config["training_config"]["N_Epoch"] = 10
    # common_config["training_config"]["validate_times"] = 5
    # ---------------------------configs for testing purpose---------------------------------#

    big_df = {}

    dataset_path = "datasets/UCI_datasets"

    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:
    # for dataname in ["boston"]:

        err_mu_dic = {}
        err_std_dic = {}

        x, y = get_uci_data(data_name= dataname, dir_name= dataset_path)

        (train_X_raw, train_Y_raw), (recal_X_raw, recal_Y_raw), (val_X, val_Y), (test_X, test_Y) = common_processor_UCI(x, y)

        for modelname in ["PureKernel","NRC", "NRC-RF", "NRC-RP", "NRC-Cov", "NRC-RF-RP"]:


            seed_list = np.arange(5)
            
            crits_dic = {}

            # train base model
            print("model: "+ modelname +" on data: "+dataname)

            for k in range(5):
                seed = seed_list[k]
                seed_all(seed)

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


                if modelname in ["NRC", "NRC-RP", "NRC-Cov"]:

                    DNN_Trainer(base_model, train_X, train_Y, val_X, val_Y, **configs["training_config"])

                elif modelname in ["NRC-RF", "NRC-RF-RP"]:

                    depth = 10

                    base_model = RandomForestRegressor(max_depth=depth, random_state=0)
                    base_model.fit(train_X.cpu().numpy(), train_Y.cpu().numpy())


                # some additional operations

                if modelname in ["PureKernel", "NRC", "NRC-RF"]:

                    reformer = None

                else:

                    n_component = 5

                    if modelname in ["NRC-RP", "NRC-RF-RP"]:

                        transformer = random_projection.GaussianRandomProjection(n_components = n_component)

                        reformer = lambda x : transformer.fit_transform(x)

                    elif modelname == "NRC-Cov":

                        temp_y = copy.deepcopy(recal_Y_raw)
                        temp_x = copy.deepcopy(recal_X_raw)


                        corr_li = np.zeros(temp_x.shape[1])

                        for i in range(temp_x.shape[1]):
                            
                            corr_li[i] = np.abs(np.corrcoef(temp_x[:, i], temp_y)[0,1])
                            
                            
                        sorted_CORR = np.sort(corr_li)

                        threshold = sorted_CORR[-n_component]
                            
                            
                        BEST_idx = np.where(corr_li >= threshold)[0]
                        if len(BEST_idx) > n_component:
                            BEST_idx = BEST_idx[:n_component]

                        reformer = lambda x : x[:, BEST_idx]

                aux_info['reformer'] = reformer

                # many choices of wid

                for wid in np.arange(1, 25, 3):


                    aux_info['wid'] = wid

                    record = NRC_testPerform_customizer(test_X, test_Y, model_name= modelname, model = base_model, \
                                                aux_info = aux_info)



                
                    if k == 0:
                        for key in record.keys():

                            crits_dic[modelname + "_w" + str(wid) +"_"+key] = []

                    for key in record.keys():

                        crits_dic[modelname + "_w" + str(wid) +"_"+key].append(record[key])


            for key in crits_dic.keys():
                err_mu_dic[key] = np.mean(crits_dic[key])
                err_std_dic[key] = np.std(crits_dic[key])


        if len(big_df) == 0:
            big_df["idxes"] = list(err_mu_dic.keys())

        big_df[dataname +"_mu"] = list(err_mu_dic.values())
        big_df[dataname + "_std"] = list(err_std_dic.values())
        


    df = pd.DataFrame.from_dict(big_df)  

    df.to_csv("experiments/EXP1/record_bin/EXP1_NRCexperiments.csv",index=False)



if __name__ == "__main__":

    run_NRC()




                









