import os
from src.dataset_bundle import FEMNIST as FEMNISTBundle
import torch
import copy
import time
import multiprocessing as mp
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from src.datasets import FEMNIST
from tqdm.auto import tqdm
from pathlib import Path

PACS_DOMAIN_LIST = ["photo", "art_painting", "cartoon", "sketch"]

def worker(root_dir, queue):
    dataset = FEMNIST(version='1.0', root_dir=root_dir, download=True)
    indices = torch.arange(len(dataset)).reshape(-1,1)
    new_metadata_array = torch.cat((dataset.metadata_array, indices), dim=1)
    ds = FEMNISTBundle(dataset)
    dataset._metadata_array = new_metadata_array
    # train_dataset = dataset.get_subset('train', transform=ds.test_transform)
    for name in ['id_val', 'val', 'id_test', 'test']:
        train_dataset = dataset.get_subset(name, transform=ds.test_transform)
        print(name)
        dataloader = get_train_loader("standard", train_dataset, batch_size=64, uniform_over_groups=None, grouper=dataset._eval_grouper, distinct_groups=True, n_groups_per_batch=1)
        for batch in tqdm(dataloader):
            # print("Current Queue Size: {}".format(queue.qsize()))
            #TODO: update PACS with indices in the metaarray
            with torch.no_grad():
                x, domain = batch[0], batch[2]
                idx = domain[:,2]
                path = dataset._input_array[idx]
                fft = torch.fft.fft2(x)
                amp, pha = torch.abs(fft).data, torch.angle(fft).data
                queue.put([idx, path, amp, pha])
    print('Worker done, exit!')
    return 'Done'

def listener(root_dir, process_num, queue):
    while True:
        obj = queue.get()
        if isinstance(obj, str) and obj == "Exit":
            print("Job done, process {} exit".format(process_num))
            return 0
        else:
            idx, path, amp, pha = obj
            root_dir = root_dir / Path("femnist_v1.0/")
            for i, idx in enumerate(idx):
                torch.save(amp[i].clone(), str((root_dir / Path(path[i])).with_suffix(".amp")))
                torch.save(pha[i].clone(), str((root_dir / Path(path[i])).with_suffix(".pha")))


if __name__ == '__main__':
    root_dir = Path("/local/scratch/a/bai116/datasets/")
    torch.multiprocessing.set_sharing_strategy('file_system')
    manager = mp.Manager()
    q = manager.Queue()
    total_processes = 2
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

