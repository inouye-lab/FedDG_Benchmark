import os
import csv
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path

from wilds.common.utils import map_to_id_array
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.celebA_dataset import CelebADataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset


class FourierSubset(WILDSSubset):
    def __getitem__(self, idx):
        x, y, metadata, amp, pha = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y, metadata, amp, pha


class FourierIwildCam(IWildCamDataset):
    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        amp, pha = self.get_fourier(idx)
        return x, y, [metadata, amp, pha]

    def get_fourier(self, idx):
        path = os.path.join(self._data_dir, 'fourier/')
        amp = torch.load(os.path.join(path, "amp_{}.pt".format(idx)))
        pha = torch.load(os.path.join(path, "pha_{}.pt".format(idx)))
        return amp, pha


class PACS(WILDSDataset):
    _dataset_name = "pacs"
    _versions_dict = {
        '1.0': {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x19f5d1758a184e13aeaea05e0954422a/contents/blob/",
            "compressed_size": "171_612_540"
            }
    }
    def __init__(
        self, 
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official"
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (224, 224)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(self.initialize_data_dir(root_dir, download))

        # The original dataset contains 7 categories. 
        if self._split_scheme == 'official':
            metadata_filename = "metadata.csv"
        else:
            metadata_filename = "{}.csv".format(self._split_scheme)
        self._n_classes = 7

        # Load splits
        df = pd.read_csv(self._data_dir / metadata_filename)
        # Filenames
        self._input_array = df['path'].values
        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df["split_id"].values
        # Y
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        # Populate metadata fields
        self._metadata_fields = ["domain", "y", "idx"]
        self._metadata_array = torch.tensor(np.stack([df['domain_remapped'].values,
                            df['y'].values, np.arange(df['y'].shape[0])], axis=1))
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['domain']))

        super().__init__(root_dir, download, self._split_scheme)


    def get_input(self, idx) -> str:
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / self._input_array[idx]
        img = Image.open(img_path).convert("RGB")

        return img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_eval(
            metric, y_pred, y_true
        )


class FourierPACS(PACS):
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        amp, pha = self.get_fourier(idx)
        return x,y,[metadata, amp, pha]
    
    def get_fourier(self, idx):
        img_path = Path(self.data_dir / self._input_array[idx])
        amp_path = img_path.with_suffix(".amp")
        pha_path = img_path.with_suffix(".pha")
        amp = torch.load(str(amp_path))
        pha = torch.load(str(pha_path))
        return amp, pha


class FEMNIST(WILDSDataset):
    _dataset_name = "femnist"
    _versions_dict = {
        '1.0': {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x7704c8584dac49d8b8c3de5d3c617c2d/contents/blob/",
            "compressed_size": "113_126_1007"
            }
    }
    def __init__(
        self, 
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official"
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (28, 28)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(self.initialize_data_dir(root_dir, download))

        # The original dataset contains 7 categories. 
        metadata_filename = "metadata.csv"
        self._n_classes = 62

        # Load splits
        df = pd.read_csv(self._data_dir / metadata_filename)
        # Filenames
        self._input_array = df['path'].values
        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df["split_id"].values
        # Y
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        # Populate metadata fields
        self._metadata_fields = ["domain", "y", "idx"]
        self._metadata_array = torch.tensor(np.stack([df['domain_remapped'].values,
                            df['y'].values, np.arange(df['y'].shape[0])], axis=1))
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['domain']))

        super().__init__(root_dir, download, self._split_scheme)


    def get_input(self, idx) -> str:
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / self._input_array[idx]
        img = Image.open(img_path).convert("L")
        return img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_eval(
            metric, y_pred, y_true
        )

class FourierFEMNIST(FEMNIST):
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        amp, pha = self.get_fourier(idx)
        return x,y,[metadata, amp, pha]
    
    def get_fourier(self, idx):
        img_path = Path(self.data_dir / self._input_array[idx])
        amp_path = img_path.with_suffix(".amp")
        pha_path = img_path.with_suffix(".pha")
        amp = torch.load(str(amp_path))
        pha = torch.load(str(pha_path))
        return amp, pha


class OfficeHome(WILDSDataset):
    _dataset_name = "office_home"
    _versions_dict = {
        '1.0': {
            "download_url": "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
            "compressed_size": "174_167_459"
            }
    }
    def __init__(
        self, 
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official"
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (224, 224)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(self.initialize_data_dir(root_dir, download))

        # The original dataset contains 7 categories. 
        if self._split_scheme == 'official':
            metadata_filename = "metadata.csv"
        else:
            metadata_filename = "{}.csv".format(self._split_scheme)
        self._n_classes = 65

        # Load splits
        df = pd.read_csv(self._data_dir / metadata_filename)
        # Filenames
        self._input_array = df['path'].values
        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}
        print(df['split'])
        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df["split_id"].values
        # Y
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        # Populate metadata fields
        self._metadata_fields = ["domain", "y", "idx"]
        self._metadata_array = torch.tensor(np.stack([df['domain_remapped'].values,
                            df['y'].values, np.arange(df['y'].shape[0])], axis=1))
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['domain']))

        super().__init__(root_dir, download, self._split_scheme)


    def get_input(self, idx) -> str:
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / self._input_array[idx]
        img = Image.open(img_path).convert("RGB")

        return img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_eval(
            metric, y_pred, y_true
        )


class FourierOfficeHome(OfficeHome):
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        amp, pha = self.get_fourier(idx)
        return x,y,[metadata, amp, pha]
    
    def get_fourier(self, idx):
        img_path = Path(self.data_dir / self._input_array[idx])
        amp_path = img_path.with_suffix(".amp")
        pha_path = img_path.with_suffix(".pha")
        amp = torch.load(str(amp_path))
        pha = torch.load(str(pha_path))
        return amp, pha


class FourierCelebA(CelebADataset):
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        amp, pha = self.get_fourier(idx)
        return x,y,[metadata, amp, pha]
    
    def get_fourier(self, idx):
        img_path = Path(self.data_dir) / "img_align_celeba" / self._input_array[idx]
        amp_path = img_path.with_suffix(".amp")
        pha_path = img_path.with_suffix(".pha")
        amp = torch.load(str(amp_path))
        pha = torch.load(str(pha_path))
        return amp, pha



class FourierCamelyon17(Camelyon17Dataset):
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        amp, pha = self.get_fourier(idx)
        return x,y,[metadata, amp, pha]
    
    def get_fourier(self, idx):
        img_path = Path(self.data_dir) / self._input_array[idx]
        amp_path = img_path.with_suffix(".amp")
        pha_path = img_path.with_suffix(".pha")
        amp = torch.load(str(amp_path))
        pha = torch.load(str(pha_path))
        return amp, pha