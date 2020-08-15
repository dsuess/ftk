import numpy as np
import torch
from typing import Callable, Any
from torch.nn.utils import rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence


__all__ = ["Normalize", "ToTensor", "Compose", "sequence_one_hot"]


class Compose:
    def __init__(self, *transforms: Callable):
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


class Normalize:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def __call__(self, data: str) -> str:
        if self.lowercase:
            data = data.lower()
        return data.strip()


DEFAULT_CHARS = "abcdefghijklmnopqrstuvwxyz"


class ToTensor:
    def __init__(
        self,
        chars: str = DEFAULT_CHARS,
        one_hot: bool = True,
        zero_is_padding: bool = True,
        dtype=torch.int64,
    ):
        assert " " not in chars
        self.chars = " " + chars if zero_is_padding else chars
        self.one_hot = one_hot
        self.dtype = dtype

    def __call__(self, data: str) -> torch.Tensor:
        """
        >>> ToTensor(one_hot=False)("abcde")
        tensor([1, 2, 3, 4, 5])
        >>> ToTensor(one_hot=False, zero_is_padding=False)("abcde")
        tensor([0, 1, 2, 3, 4])
        >>> ToTensor(chars="abcde", one_hot=True)("ace")
        tensor([[0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1]])
        """
        valid_chars = self.chars
        try:
            chars = torch.tensor([valid_chars.index(c) for c in data])
        except ValueError:
            raise ValueError(f"chars='{valid_chars}' incomplete for word={data}")

        if not self.one_hot:
            return chars.to(self.dtype)

        result = torch.zeros(len(chars), len(valid_chars), dtype=self.dtype)
        # vals = torch.ones(len(chars), dtype=result.dtype)
        result.scatter_(1, chars[:, None], 1)
        return result


def collate_sequences(batch):
    # order by sequence that get passed through the model
    xs, *_ = zip(*batch)
    order = torch.argsort(torch.tensor([len(x) for x in xs]), descending=True)
    return tuple(
        rnn_utils.pack_sequence([tensors[i] for i in order]) for tensors in zip(*batch)
    )


def sequence_one_hot(seq: PackedSequence, size: int) -> PackedSequence:
    data = torch.zeros(seq.data.size(0), size, dtype=seq.data.dtype)
    data.scatter_(1, seq.data[:, None], 1)
    return PackedSequence(
        data, seq.batch_sizes, seq.sorted_indices, seq.unsorted_indices,
    )


class KeyboardNoise:
    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

    def __init__(self, characters: str, error_rate=0.2):
        eye = torch.eye(len(characters), dtype=torch.float32)
        error_mat = self.build_transition_matrix(characters, self.layout)
        self.transition_matrix = (1 - error_rate) * eye + error_rate * error_mat

    @staticmethod
    def build_transition_matrix(characters, layout):
        max_len = max(len(s) for s in layout)
        padded = np.array(
            [[""] + list(s) + [""] * (max_len - len(s) + 1) for s in layout]
        )
        pads = np.array([""] * (max_len + 2))[None]
        padded = np.concatenate([pads, padded, pads], axis=0)

        transition_matrix = np.zeros((len(characters),) * 2, dtype=np.float32)

        directions = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])

        for idx, char in enumerate(characters):
            (loc,) = np.argwhere(padded == char)
            for d in directions:
                neighboar_char = padded[tuple(loc + d)]
                if neighboar_char:
                    idx_ = characters.index(neighboar_char)
                    transition_matrix[idx, idx_] += 1

        transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
        return torch.tensor(transition_matrix)

    def __call__(self, word):
        """
        >>> transform = KeyboardNoise('abcdefghijklmnopqrstuvwxyz', error_rate=1.0)
        >>> word = torch.tensor([11, 12, 13, 14])
        >>> all(transform(word) != word)
        True

        >>> transform = KeyboardNoise('abcdefghijklmnopqrstuvwxyz', error_rate=0.0)
        >>> word = torch.tensor([11, 12, 13, 14])
        >>> all(transform(word) == word)
        True

        """
        probabilities = self.transition_matrix[word]
        noisy = torch.multinomial(probabilities, num_samples=1)
        return noisy.squeeze(dim=1)

