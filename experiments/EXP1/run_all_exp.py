from NRC.utils import seed_all
from experiments.data_utils import get_uci_data, common_processor_UCI


if __name__ == "__main__":

    dataset_path = "datasets/UCI_datasets"

    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:

        x, y = get_uci_data(data_name= dataname, dir_name= dataset_path)

        (train_X, train_Y), (recal_X, recal_Y), (val_X, val_Y), (test_X, test_Y) = common_processor_UCI(x, y)

        for modelname in ["HNN", "MCDrop", "DeepEnsemble", "CCL", "ISR", "DGP", "MMD", "MAQR", "CQR", "OQR", "NRC", "NRC-RF", "NRC-RP", "NRC-Cov"]:



            for seed in range(5):

                seed_all(seed)

                print(seed)









