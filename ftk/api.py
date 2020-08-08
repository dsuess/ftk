import functools as ft
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import LightningModule
from ftk.data import WordDataset, transforms, sequence_one_hot
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def filter_word_list(words, chars):
    return [word for word in words if set(word).issubset(chars) and len(word) > 0]


class SimpleModel(LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.datafile = "data/words/tinyshakespeare.txt"
        self.characters = "abcdefghijklmnopqrstuvwxyz"
        self.rnn_layer = nn.RNN(len(self.characters), hidden_size=32)
        self.fc_layer = nn.Linear(32, len(self.characters))
        self.hparams = hparams if hparams is not None else dict()

    def train_dataloader(self):
        with open(self.datafile) as buf:
            words = filter_word_list(buf.read().split("\n"), self.characters)

        dataset = WordDataset(
            words,
            data_preparation=transforms.Compose(
                transforms.Normalize(),
                transforms.ToTensor(
                    chars=self.characters,
                    zero_is_padding=False,
                    one_hot=False,
                    dtype=torch.int64,
                ),
            ),
        )
        sampler = RandomSampler(dataset, replacement=True, num_samples=4096)

        return DataLoader(
            dataset,
            sampler=sampler,
            num_workers=0,
            batch_size=self.hparams.get("batch_size", 32),
            collate_fn=transforms.collate_sequences,
        )

    @staticmethod
    def evaluate_prediction(prediction: PackedSequence, target: PackedSequence):
        result = dict()
        with torch.no_grad():
            char_pred = prediction.data.argmax(dim=1)
            result["char_accuracy"] = (1.0 * (char_pred == target.data)).mean()

            char_pred = PackedSequence(
                char_pred,
                prediction.batch_sizes,
                prediction.sorted_indices,
                prediction.unsorted_indices,
            )
            target_u, _ = pad_packed_sequence(
                target, padding_value=-1, batch_first=True
            )
            pred_u, _ = pad_packed_sequence(
                char_pred, padding_value=-1, batch_first=True
            )
            is_invalid = (target_u < 0) * (pred_u < 0)
            correct_word = torch.all((target_u == pred_u) + is_invalid, dim=1)
            result["word_accuracy"] = (1.0 * correct_word).mean()

        return result

    def training_step(self, batch, _):
        augmented, original = batch
        # One-hot encoding
        augmented = sequence_one_hot(augmented, len(self.characters)).to(torch.float32)
        logits = self.forward(augmented)
        loss = nn.functional.cross_entropy(logits.data, original.data, reduce="mean")
        log = self.evaluate_prediction(logits, original)
        return {"loss": loss, "log": log}

    def forward(self, x):
        y, _ = self.rnn_layer(x)
        y = self.fc_layer(y.data)
        return PackedSequence(y, x.batch_sizes, x.sorted_indices, x.unsorted_indices)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.get("lr", 0.01))
