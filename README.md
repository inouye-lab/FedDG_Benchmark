# FedDG_Benchmark
Welcome! This is the code repo for paper "[Benchmarking Algorithms for Federated Domain Generalization]()".

# Available methods
* FedAvg
* IRM
* REx
* Fish
* MMD
* DeepCoral
* GroupDRO
* FedProx
* Scaffold
* FedDG
* FedADG
* FedSR
* FedGMA
# Available datasets
All datasets derived from [Wilds](https://wilds.stanford.edu/) Datasets. We also implement [femnist](https://leaf.cmu.edu/), [PACS](https://arxiv.org/abs/2007.01434) and [OfficeHome](https://arxiv.org/abs/2007.01434) datasets.

# Quick Start
```bash
python main.py --config_file config/ERM/pacs/fl_0.json
```
