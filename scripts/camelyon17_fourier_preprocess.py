import os

from src.dataset_bundle import Camelyon17 as Camelyon17Bundle
import torch
import copy
import time
import multiprocessing as mp
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from src.datasets import PACS
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np


def worker(root_dir, queue):
    dataset = get_dataset(dataset='camelyon17', root_dir=root_dir, download=True)
    indices = torch.arange(len(dataset)).reshape(-1,1)
    new_metadata_array = torch.cat((dataset.metadata_array, indices), dim=1)
    
    ds = Camelyon17Bundle(dataset, feature_dimension=1024)
    dataset._metadata_array = new_metadata_array
    input_array = np.array(dataset._input_array)
    for split in ['train','id_val', 'val','test']:
        train_dataset = dataset.get_subset(split, transform=ds.test_transform)
        dataloader = get_train_loader("standard", train_dataset, batch_size=64, uniform_over_groups=None, grouper=dataset._eval_grouper, distinct_groups=True, n_groups_per_batch=1)
        for batch in tqdm(dataloader):
            # print("Current Queue Size: {}".format(queue.qsize()))
            with torch.no_grad():
                x, idx = batch[0], batch[2][:,-1]
                idx = idx.numpy().astype(int)
                path = input_array[idx]
                fft = torch.fft.fft2(x)
                amp, pha = torch.abs(fft).data, torch.angle(fft).data

                queue.put([path, amp, pha])
    print('Worker done, exit!')
    return 'Done'

def listener(root_dir, process_num, queue):
    root_dir = root_dir / Path("camelyon17_v1.0/")
    while True:
        obj = queue.get()
        if isinstance(obj, str) and obj == "Exit":
            print("Job done, process {} exit".format(process_num))
            return 0
        else:
            path, amp, pha = obj
            for i, p in enumerate(path):
                torch.save(amp[i].clone(), str((root_dir / Path(p)).with_suffix(".amp")))
                torch.save(pha[i].clone(), str((root_dir / Path(p)).with_suffix(".pha")))


if __name__ == '__main__':
    root_dir = "/local/scratch/a/bai116/datasets/"    
    manager = mp.Manager()
    q = manager.Queue()
    total_processes = 6
    pool = mp.Pool(total_processes)
    watchers = []
    print("starting listener")
    for i in range(total_processes-1):
        watcher = pool.apply_async(listener, (root_dir, i, q))
        watchers.append(watcher)
    print("worker")
    job = pool.apply_async(worker, (root_dir, q))
    job.get()
    for i in range(total_processes-1):
        q.put("Exit")
    for watcher in watchers:
        watcher.get()
    if q.empty():
        pool.close()
    else:
        print(q.size())

