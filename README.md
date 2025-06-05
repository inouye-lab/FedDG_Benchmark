# Federated Domain Generalization Benchmark

This is the git repo of [Benchmarking Algorithms for Domain Generalization in Federated Learning](https://openreview.net/forum?id=wprSv7ichW).

## Available methods
* FedAvg
* IRM
* REx
* Fish
* MMD
* DeepCoral
* GroupDRO
* FedProx
* Scaffold
* AFL
* FedDG
* FedADG
* FedSR
* FedGMA

# Environment preparation
```
conda create  --name <env> --file requirements.txt
```

## Prepare Datasets
All datasets derived from [Wilds](https://wilds.stanford.edu/) Datasets. We also implement [femnist](https://leaf.cmu.edu/) and [PACS](https://arxiv.org/abs/2007.01434) datasets.
Here's the backup link if ones in the code do not work. [pacs](https://drive.google.com/file/d/1qIcjspcQrbx6iDDDE1kYz3ccGjyr5485/view?usp=sharing) and [femnist](https://drive.google.com/file/d/12s5rMqVWpw7x1JFYtKQM1cisBgHot01M/view?usp=sharing)

### Preparing metadata.csv and RELEASE_v1.0.txt
For PACS and FEMNIST dataset, please put 
```
resources/femnist_v1.0/* 
```
and 
```
resources/pacs_v1.0/* 
```
into your dataset directory.

### Preparing fourier transformation
Some methods require fourier transformation. To accelerate training, we should prepare the transformation data in advance. Please first load the scripts in the scripts path. Note: Please config the root_path in the script.

### Prepare WanDB.
First, please register an account on [WanDB](https://wandb.ai/). Then in wandb_env.py, fill in the entity name and the project name (You could name another name for your project).

### Setup the global config.

## Run Experiments
To run the experiments, simply prepare your config file $config_path, and run
```
python main.py --config_file $config_path
```
For example, to run fedavg-erm with centralized learning on iwildcam, run
```
python main.py --config_file ./config/ERM/iwildcam/centralized.json
```
## Run Sweep
To sweep over hyperparameters, simply prepare your config file $sweep_path, and run
```
python sweep.py --sweep_config $config_path
```
For example, to sweep the learning rate of fedavg-erm on CelebA, run
```
python sweep.py --sweep_config sweep/hparam_search/celeba/erm_hs.json
```

# Contributing New Dataset and Methods
## Dataset
We support all datasets derived from [WILDSDataset](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/wilds_dataset.py) in Wilds Benchmark. You need to modify the main.py file to import the class manually. If the dataset is image dataset and will run FedDG, please preprocess the dataset. You can follow our demo code in script/. After that, you will need to create a fourier version of the dataset class just like we did in src/datasets.py. Don't forget to edit the main.py to import this class as well.

## Dataset Bundle
This object is a code configuration. We put dataset related configuration like the domain fields, resolution, data transforms, loss functions, number of classes, etc, here. Please read the code src/dataset_bundle.py for reference.
> ⚠️ **Note:** the bundle class should be named exactly the same as the dataset class name.

## Methods
src/server.py and src/client.py contain the server side and client side method respectively. You could write your own server and client class as long as the interface with main.py is compatible. Or you can derive from our classes.

### Server
All our methods in the server side are derived from the FedAvg class. This is a basic class define the behaviour of the client management, model transmition and collection, data aggreagation, client sampling, etc. Please derive from FedAvg and include your method. For instance, if your method has a different aggregation rule than FedAvg, just derive from FedAvg and reimplement aggregate method.

### Client
All our methods in the client side are derived from the ERM class. fit method defines the overall training bewteen two communications. process_batch defines the pre-processing of one batch of data sample, and step defines one step of the objective updates. 

Most of time, you only need to re-implement the step method. In your method follows different procedure compared to ERM, you may need to reimplement fit method. 
> ⚠️ **Note:** Don't forget to derive the __init__ method and include your own hyperparameter initialization.
