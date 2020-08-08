import torch
from typing import Union, Callable, List, Tuple, Optional
from . import transforms


__all__ = ["WordDataset"]


class WordDataset(torch.utils.data.Dataset):
    """

    >>> ds = WordDataset(["hello", "world"], data_preparation=lambda x: x, augmentation=lambda x: x)
    >>> len(ds)
    2
    >>> ds[0]
    ('hello', 'hello')
    >>> ds[1]
    ('world', 'world')

    >>> ds = WordDataset(["hello", "world"], data_preparation=lambda x: x, augmentation=lambda x: x.replace('e', '3'))
    >>> len(ds)
    2
    >>> ds[0]
    ('h3llo', 'hello')
    >>> ds[1]
    ('world', 'world')
    """

    def __init__(
        self,
        *data: List[str],
        data_preparation: Callable[[str], torch.Tensor],
        augmentation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        self.data: List[str] = sum(data, [])
        self.data_preparation = data_preparation
        self.augmentation = augmentation if augmentation is not None else lambda x: x

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        tensors = self.data_preparation(data)
        return self.augmentation(tensors), tensors
