import torch
from ...utils import count_tensor_params
from .base_transforms import Transform, DictTransform
from typing import Dict

class MinMaxNormalizer(Transform):
    """
    MinMaxNormalizer normalizes data to a specified range, defaulting to [0,1].
    """
    def __init__(self, min_val=None, max_val=None, eps=1e-7, dim=None, mask=None, range_min=0, range_max=1):
        """
        min_val : torch.tensor or None
            If None, will compute from dataset.
        max_val : torch.tensor or None
        eps : float, default is 1e-7
            To prevent division by zero.
        dim : int list, default is None
            If not None, dimensions to compute min/max over.
        range_min, range_max : float
            The target range for normalization.
        """
        super().__init__()

        self.register_buffer("min_val", min_val)
        self.register_buffer("max_val", max_val)
        self.register_buffer("mask", mask)

        self.eps = eps
        self.range_min = range_min
        self.range_max = range_max

        if min_val is not None:
            self.ndim = min_val.ndim
        if isinstance(dim, int):
            dim = [dim]  # Ensure dim is a list
        self.dim = dim
        self.n_elements = 0

    def fit(self, data_batch):
        """Computes min and max values from the dataset."""
        self.update_min_max(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        """Incrementally updates min/max values in batches."""
        if 0 in list(data_batch.shape):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            if self.n_elements:
                self.incremental_update_min_max(samples)
            else:
                self.update_min_max(samples)
            count += batch_size

    def update_min_max(self, data_batch):
        """Computes min/max over the dataset."""
        self.ndim = data_batch.ndim  
        if self.mask is None:
            self.n_elements = count_tensor_params(data_batch, self.dim)
            self.min_val = torch.amin(data_batch, dim=self.dim, keepdim=True)  #  Fixed
            self.max_val = torch.amax(data_batch, dim=self.dim, keepdim=True)  #  Fixed
        else:
            batch_size = data_batch.shape[0]
            dim = [i - 1 for i in self.dim if i]
            shape = [s for i, s in enumerate(self.mask.shape) if i not in dim]
            self.n_elements = torch.count_nonzero(self.mask, dim=dim) * batch_size
            self.min_val = torch.full(shape, float("inf"))
            self.max_val = torch.full(shape, float("-inf"))
            masked_data = data_batch * self.mask
            self.min_val[self.mask == 1] = torch.amin(masked_data, dim=dim, keepdim=True)
            self.max_val[self.mask == 1] = torch.amax(masked_data, dim=dim, keepdim=True)

    def incremental_update_min_max(self, data_batch):
        """Incrementally updates min/max values with new batch."""
        self.min_val = torch.minimum(self.min_val, torch.amin(data_batch, dim=self.dim, keepdim=True))
        self.max_val = torch.maximum(self.max_val, torch.amax(data_batch, dim=self.dim, keepdim=True))

    def transform(self, x):
        """Applies Min-Max Normalization to transform data to [range_min, range_max]."""
        return ((x - self.min_val) / (self.max_val - self.min_val + self.eps)) * \
               (self.range_max - self.range_min) + self.range_min

    def inverse_transform(self, x):
        """Reverses Min-Max Normalization to recover original values."""
        return ((x - self.range_min) / (self.range_max - self.range_min)) * \
               (self.max_val - self.min_val + self.eps) + self.min_val

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.min_val = self.min_val.cuda()
        self.max_val = self.max_val.cuda()
        return self

    def cpu(self):
        self.min_val = self.min_val.cpu()
        self.max_val = self.max_val.cpu()
        return self

    def to(self, device):
        self.min_val = self.min_val.to(device)
        self.max_val = self.max_val.to(device)
        return self

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Creates normalizer instances from a dataset."""
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances

