import os
from src.dataset_bundle import IWildCam
import torch
import copy
import time
import multiprocessing as mp
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from tqdm.auto import tqdm

def worker(root_dir, queue):
    dataset = get_dataset(dataset='iwildcam', root_dir=root_dir, download=True)
    indices = torch.arange(len(dataset)).reshape(-1,1)
    new_metadata_array = torch.cat((dataset.metadata_array, indices), dim=1)
    dataset._metadata_array = new_metadata_array
    ds = IWildCam(dataset)
    train_dataset = dataset.get_subset('train', transform=ds.test_transform)
    dataloader = get_train_loader("standard", train_dataset, batch_size=64, uniform_over_groups=None, grouper=ds.grouper, distinct_groups=True, n_groups_per_batch=1)
    for batch in tqdm(dataloader):
        # print("Current Queue Size: {}".format(queue.qsize()))
        with torch.no_grad():
            x, indices = batch[0], batch[2][:,-1]
            fft = torch.fft.fft2(x)
            amp, pha = torch.abs(fft).data, torch.angle(fft).data
            queue.put([indices, amp, pha])
    print('Worker done, exit!')
    return 'Done'

def listener(root_dir, process_num, queue):
    while True:
        obj = queue.get()
        if isinstance(obj, str) and obj == "Exit":
            print("Job done, process {} exit".format(process_num))
            return 0
        else:
            indices, amp, pha = obj
            path = os.path.join(root_dir, 'iwildcam_v2.0/fourier/')
            os.makedirs(path, exist_ok = True)
            for i, idx in enumerate(indices):
                torch.save(amp[i].clone(), os.path.join(path, 'amp_{}.pt'.format(idx)))
                torch.save(pha[i].clone(), os.path.join(path, 'pha_{}.pt'.format(idx)))


if __name__ == '__main__':
    root_dir = "/local/scratch/a/shared/datasets/"
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

