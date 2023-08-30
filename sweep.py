import wandb
import argparse
import json

from wandb_env import *


parser = argparse.ArgumentParser(description='FedDG Benchmark Sweep')
parser.add_argument('--sweep_config', help='sweep config file')
args = parser.parse_args()
with open(args.sweep_config) as sf:
        hparam = json.load(sf)

if len(hparam['parameters']['dataset']['values']) > 1:
    raise ValueError('could not sweep over multiple dataset')
elif len(hparam['parameters']['dataset']['values']) == 0:
    raise ValueError('Must contain one dataset')
wandb_project = WANDB_PROJECT + '_' + hparam['parameters']['dataset']['values'][0]
sweep_id = wandb.sweep(sweep=hparam, project=wandb_project, entity=WANDB_ENTITY)

wandb.agent(sweep_id, count=8)