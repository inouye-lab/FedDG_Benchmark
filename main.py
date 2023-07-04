import logging
import time
import datetime
import gc
import argparse
import torch
import torch.cuda
from src.server import *
from src.client import *
import src.datasets as my_datasets
# from dataclasses import dataclass
from src.splitter import *
from src.utils import *
from src.dataset_bundle import *
from wilds.common.data_loaders import get_eval_loader
from wilds import get_dataset
from src.models import ResNet

"""
The main file function:
1. Load the hyperparameter dict.
2. Initialize logger
3. Initialize data (preprocess, data splits, etc.)
4. Initialize clients. 
5. Initialize Server.
6. Register clients at the server.
7. Start the server.
"""

def main(args):
    config_file = args.config_file
    with open(config_file) as fh:
        hparam = json.load(fh)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_config = hparam["server"]
    client_config = hparam["client"]
    data_config = hparam["dataset"]
    global_config = hparam["global"]
    seed = global_config['seed']
    exp_id = global_config['id']
    data_path = global_config['data_path']
    set_seed(seed)
    if not os.path.exists(data_path + "opt_dict/"): os.makedirs(data_path + "opt_dict/")
    if not os.path.exists(data_path + "models/"): os.makedirs(data_path + "models/")
    # 1. Preprocess some hyperparameters
    data_config["num_shards"] = global_config["num_clients"]
    client_config["batch_size"] = global_config["batch_size"]
    data_config["seed"] = global_config["seed"]
     # 2. Initialize logger.
    # 2.1 modify log_path to contain current time
    log_path = os.path.join(global_config["log_path"], str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")[:-3]))
    os.makedirs(log_path)
        # 2.2 set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_path, "FL.log"),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")
    
    # 2.3 display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    logging.info(message)
    logging.info(hparam)

    # initialize data
    num_shards = data_config['num_shards']
    iid = data_config['iid']
    root_dir = data_config["data_path"]
    if global_config['dataset_name'].lower() == 'pacs':
        dataset = my_datasets.PACS(version='1.0', root_dir=root_dir, download=True)
    elif global_config['dataset_name'].lower() == 'officehome':
        dataset = my_datasets.OfficeHome(version='1.0', root_dir=root_dir, download=True, split_scheme=data_config["split_scheme"])
    elif global_config['dataset_name'].lower() == 'femnist':
        dataset = my_datasets.FEMNIST(version='1.0', root_dir=root_dir, download=True)
    else:
        dataset = get_dataset(dataset=global_config["dataset_name"].lower(), root_dir=root_dir, download=True)
    # if server_config['algorithm'] == "FedDG":
    #     # make it easier to hash fourier transformation
    #     indices = torch.arange(len(dataset)).reshape(-1,1)
    #     new_metadata_array = torch.cat((dataset.metadata_array, indices), dim=1)
    #     dataset._metadata_array = new_metadata_array
    if client_config['algorithm'] == "FedSR":
        ds_bundle = eval(global_config["dataset_name"])(dataset, global_config["feature_dimension"], probabilistic=True)
    else:
        if global_config['dataset_name'].lower() == 'py150' or global_config['dataset_name'].lower() == 'civilcomments':
            ds_bundle = eval(global_config["dataset_name"])(dataset, probabilistic=False)
        else:
            ds_bundle = eval(global_config["dataset_name"])(dataset, global_config["feature_dimension"], probabilistic=False)
    if server_config['algorithm'] == "FedDG":
        if global_config["dataset_name"].lower() == "iwildcam":
            dataset = my_datasets.FourierIwildCam(root_dir=root_dir, download=True)
            total_subset = dataset.get_subset('train', transform=ds_bundle.test_transform)
        elif global_config["dataset_name"].lower() == "pacs":
            dataset = my_datasets.FourierPACS(root_dir=root_dir, download=True, split_scheme=data_config["split_scheme"])
            total_subset = dataset.get_subset('train', transform=ds_bundle.test_transform)
        else:
            raise NotImplementedError
    else:
        total_subset = dataset.get_subset('train', transform=ds_bundle.train_transform)

    try:
        in_test_dataset = dataset.get_subset('id_test', transform=ds_bundle.test_transform)
    except ValueError:
        in_test_dataset, total_subset = RandomSplitter(ratio=0.2, seed=seed).split(total_subset)
        in_test_dataset.transform = ds_bundle.test_transform
        total_subset.transform = ds_bundle.train_transform

    lodo_validation_dataset = dataset.get_subset('val', transform = ds_bundle.test_transform)
    try:
        in_validation_dataset = dataset.get_subset('id_val',transform = ds_bundle.test_transform)
    except ValueError:
        in_validation_dataset, in_test_dataset = RandomSplitter(ratio=0.5, seed=seed).split(total_subset)
        in_validation_dataset.transform = ds_bundle.test_transform
        in_test_dataset.transform = ds_bundle.test_transform
    out_test_dataset = dataset.get_subset('test', transform=ds_bundle.test_transform)
    
    out_test_dataloader = get_eval_loader(loader='standard', dataset=out_test_dataset, batch_size=global_config["batch_size"])
    in_test_dataloader = get_eval_loader(loader='standard', dataset=in_test_dataset, batch_size=global_config["batch_size"])
    lodo_validation_dataloader = get_eval_loader(loader='standard', dataset=lodo_validation_dataset, batch_size=global_config["batch_size"])
    in_validation_dataloader = get_eval_loader(loader='standard', dataset=in_validation_dataset, batch_size=global_config["batch_size"])
    
    sampler = RandomSampler(total_subset, replacement=True)
    global_dataloader = DataLoader(total_subset, batch_size=global_config["batch_size"], sampler=sampler)
    # # DS
    # out_test_dataset, test_train = RandomSplitter(ratio=0.5, seed=seed).split(out_test_dataset)
    # out_test_dataset.transform = ds_bundle.test_transform
    # out_test_dataloader = get_eval_loader(loader='standard', dataset=out_test_dataset, batch_size=global_config["batch_size"])
    # if global_config['cheat']:
    #     total_subset = concat_subset(total_subset, test_train)
    # training_datasets = [total_subset]
    print(len(total_subset), len(in_validation_dataset), len(lodo_validation_dataset), len(in_test_dataset), len(out_test_dataset))

    if num_shards == 1:
        training_datasets = [total_subset]
    elif num_shards > 1:
        training_datasets = NonIIDSplitter(num_shards=num_shards, iid=iid, seed=seed).split(dataset.get_subset('train'), ds_bundle.groupby_fields, transform=ds_bundle.train_transform)
    else:
        raise ValueError("num_shards should be greater or equal to 1, we got {}".format(num_shards))

    # initialize client
    clients = []
    for k in tqdm(range(global_config["num_clients"]), leave=False):
        client = eval(client_config["algorithm"])(seed, exp_id, k, device, training_datasets[k], ds_bundle, client_config)
        clients.append(client)
    message = f"successfully initialize all clients!"
    logging.info(message)
    del message; gc.collect() 

    # initialize server (model should be initialized in the server. )
    central_server = eval(server_config["algorithm"])(seed, exp_id, device, ds_bundle, server_config)
    if server_config['algorithm'] == "FedDG":
        central_server.set_amploader(global_dataloader)
    central_server.setup_model(args.resume_file, args.start_epoch)
    central_server.register_clients(clients)
    central_server.register_testloader({
        "in_val": in_validation_dataloader, 
        "lodo_val": lodo_validation_dataloader, 
        "in_test": in_test_dataloader, 
        "out_test": out_test_dataloader})
    # do federated learning
    central_server.fit()
    
    # bye!
    message = "...done all learning process!\n...exit program!"
    logging.info(message)
    time.sleep(3)
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MAT Training')
    parser.add_argument('--config_file', help='config file')
    parser.add_argument('--resume_file', default=None)
    parser.add_argument('--start_epoch', default=0)
    args = parser.parse_args()
    main(args)
