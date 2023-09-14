from typing import Sequence, List, Iterator

import numpy as np

from capbc.utils import optional_import

torch, has_torch = optional_import("torch")


class DataStore:
    """A data class that can be used to store data in a variety of formats.
    """
    def __init__(self, data: Sequence):
        """Constructor.

        Args:
            data (Sequence): data to store.
        """
        self.data = data

    def __getitem__(self, idx: int):
        """Get the data at the given index.

        Args:
            idx (int): index of the data to get.

        Returns:
            data at the given index.
        """
        return self.data[idx]

    def __len__(self) -> int:
        """Get the length of the data.

        Returns:
            int: length
        """
        return len(self.data)

    def __iter__(self) -> Iterator:
        """ Get an iterator over the data.

        Returns:
            Iterator: data iterator.
        """
        return iter(self.data)

    def __str__(self) -> str:
        """Get a string representation of the data.

        Returns:
            str: string representation.
        """
        return str(self.data)

    @property
    def shape(self):
        """ Get the shape of the data.
        """
        if isinstance(self.data, list):
            raise ValueError(
                "Cannot get shape of list. Maybe you want to cast data into a "
                "different type?"
            )
        return self.data.shape

    def cast(self, dclass, dtype=None):
        """Cast self.data to the given dtype

        Args:
            dclass: data class. choices = [np.ndarray, torch.Tensor, list]
            dtype: data type. only valid in np.ndarray and torch.Tensor
        """
        if dclass is list:
            self.data = self.as_list()
        elif dclass is np.ndarray:
            self.data = self.as_numpy(dtype)
        elif has_torch and dclass is torch.Tensor:
            self.data = self.as_tensor(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def as_list(self) -> List:
        """Convert the data to a list.

        Returns:
            List: data as a list.
        """
        if isinstance(self.data, np.ndarray):
            data = self.data.tolist()
        elif has_torch and isinstance(self.data, torch.Tensor):
            data = self.data.tolist()
        else:
            data = self.data
        return data

    def as_numpy(self, dtype: np.dtype = None) -> np.ndarray:
        """Convert the data to a numpy array.

        Args:
            dtype (np.dtype, optional): data type. Defaults to None.

        Returns:
            np.ndarray: data as a numpy array.
        """
        if isinstance(self.data, (list, tuple)):
            data = np.asarray(self.data, dtype)
        elif has_torch and isinstance(self.data, torch.Tensor):
            data = self.data.detach().cpu().numpy()
            data = np.astype(data, dtype) if dtype else data
        else:
            data = self.data
        return data

    def as_tensor(self, dtype=None):
        """Convert the data to a torch tensor.

        Args:
            dtype (torch.dtype, optional): data type. Defaults to None.

        Raises:
            RuntimeError: if torch is not available

        Returns:
            torch.Tensor: data as a torch tensor.
        """
        if not has_torch:
            raise RuntimeError("torch is not installed")

        data = torch.tensor(self.data)
        data = torch.to(dtype) if dtype else data
        return data

    def to(self, device):
        raise NotImplementedError
