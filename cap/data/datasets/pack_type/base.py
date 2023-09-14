# Copyright (c) Changan Auto. All rights reserved.

from abc import ABC

__all__ = ["PackType"]


class PackType(ABC):
    """Data type interface class."""

    def open(self):
        """Open the data file."""
        pass

    def close(self):
        """Close the data file."""
        pass

    def write(self, idx: int, record: bytes):
        """Write record into data file by idx."""
        pass

    def read(self, idx):
        """Read the idx-th data."""
        pass

    def reset(self):
        """Reset the data file operator."""
        pass

    def get_keys(self):
        """Get keys for read."""
        pass

    def __del__(self):
        """Recycle resources."""
        self.close()
