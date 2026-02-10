"""Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.


We adapt shape suffixes style when working with tensors.
See https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd.

Dimension key:

b: batch size
l: max sequence length
c: number of classes
v: vocabulary size

For example,

feature_b_l means a tensor of shape (b, l) == (batch_size, max_sequence_length).
length_1 means a tensor of shape (1) == (1,).
loss means a tensor of shape (). You can retrieve the loss value with loss.item().
"""

import argparse
import os
from collections import Counter
from pprint import pprint
from typing import Dict, List, Tuple
import re

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


class Tokenizer:
    # The index of the padding embedding.
    # This is used to pad variable length sequences.
    TOK_PADDING_INDEX = 0
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    def _pre_process_text(self, text: str) -> List[str]:
        # TODO: Implement this! Expected # of lines: 5~10
        return [
            t.lower() for t in re.split(r'[\s\W]+', text) if t and t.lower() not in Tokenizer.STOP_WORDS
        ]

    def __init__(self, data: List[DataPoint], max_vocab_size: int = None):
        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]
        # offset because padding index is 0
        self.token2id = {t: (i + 1) for i, t in enumerate(tokens)}
        self.token2id["<PAD>"] = Tokenizer.TOK_PADDING_INDEX
        # self.token2id["<UNK>"] = len(tokens) + 1
        self.id2token = {i: t for t, i in self.token2id.items()}
        # self.unk_id = self.token2id["<UNK>"]

    def tokenize(self, text: str) -> List[int]:
        # TODO: Implement this! Expected # of lines: 5~10
        return [self.token2id.get(token, self.TOK_PADDING_INDEX) for token in self._pre_process_text(text)]


def get_label_mappings(
    data: List[DataPoint],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Reads the labels file and returns the mapping."""
    labels = sorted(list(set([d.label for d in data])))
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return label2id, id2label


class BOWDataset(Dataset):
    def __init__(
        self,
        data: List[DataPoint],
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        max_length: int = 100,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a single example as a tuple of torch.Tensors.
        features_l: The tokenized text of example, shaped (max_length,)
        length: The length of the text, shaped ()
        label: The label of the example, shaped ()

        All of have type torch.int64.
        """
        dp: DataPoint = self.data[idx]
        tokenized_input = self.tokenizer.tokenize(dp.text)
        len_tokenized_input_before_padding = min(len(tokenized_input), self.max_length)
        if len(tokenized_input) > self.max_length:
            tokenized_input = tokenized_input[:self.max_length]
        elif len(tokenized_input) < self.max_length:
            tokenized_input += [Tokenizer.TOK_PADDING_INDEX] * (self.max_length - len(tokenized_input))
        # if dp.label is None:
        #     print(f"Label for data point {dp.id} is None")
        #     print(f"dp: {dp}")

        return (
            torch.tensor(tokenized_input),
            torch.tensor(len_tokenized_input_before_padding),
            torch.tensor(self.label2id[dp.label] if dp.label is not None else 0),
        )
        # TODO: Implement this! Expected # of lines: ~20


class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification."""

    EMBEDDING_DIM = 512

    def __init__(self, vocab_size: int, num_classes: int, padding_index: int):
        """Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding = torch.nn.Embedding(
            self.vocab_size, self.EMBEDDING_DIM, padding_idx=self.padding_index
        )
        self.fc1 = nn.Linear(self.EMBEDDING_DIM, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, self.num_classes)
        self.dropout = nn.Dropout(0.25)

        # TODO: Implement this!

    def forward(
        self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Inputs:
            input_features_b_l (tensor): Input data for an example or a batch of examples.
            input_length (tensor): The length of the input data.

        Returns:
            output_b_c: The output of the model.
        """
        x_embedding = self.embedding(input_features_b_l)

        x = torch.sum(x_embedding, dim=1)
        x = x / input_length_b.clamp(min=1).unsqueeze(1).float()

        # print(f"x_embedding:", x_embedding.shape)

        # shape_x1 = x_embedding.shape[-1]
        # shape_x2 = x_embedding.shape[-2]
        # print(f"{shape_x1=}, {shape_x2=}")
        # x = x_embedding.reshape(shape_x1 * shape_x2)

        # print(f"x before fc1:", x.shape)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        # print(f"x before fc2:", x.shape)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        x = self.dropout(x)

        x = torch.relu(self.fc4(x))
        x = self.dropout(x)

        # print(f"x before fc3:", x.shape)
        x = self.fc5(x)

        # print(f"x before output layer:", x.shape)
        # x = torch.softmax(x, dim=1)
        # print("sum of x:", torch.sum(x))
        return x

        # TODO: Implement this!


class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, data: BOWDataset) -> List[int]:
        """Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.

        """
        self.model.eval()
        with torch.no_grad():
            all_predictions = []
            dataloader = DataLoader(data, batch_size=128, shuffle=False)
            for features_b_l, lengths_b, labels_b in dataloader:
                output_b_c = self.model(features_b_l, lengths_b)
                output_b_c = torch.softmax(output_b_c, dim=1)
                # print("probability sum:", torch.sum(output_b_c, dim=1))
                all_predictions.extend(output_b_c.argmax(dim=-1).tolist())
            return all_predictions
        # TODO: Implement this!

    def evaluate(self, data: BOWDataset) -> float:
        """Evaluates the model on a dataset.

        Inputs:
            data: The dataset to evaluate on.

        Returns:
            The accuracy of the model.
        """
        # TODO: Implement this!
        self.model.eval()
        with torch.no_grad():
            all_predictions = []
            dataloader = DataLoader(data, batch_size=128, shuffle=False)
            labels = []
            for features_b_l, lengths_b, labels_b in dataloader:
                output_b_c = self.model(features_b_l, lengths_b)
                # print("output_b_c shape:", output_b_c.shape)
                # print("probability sum:", torch.sum(output_b_c, dim=1))
                all_predictions.extend(output_b_c.argmax(dim=-1).tolist())
                labels.extend(labels_b.tolist())
            # print(f"{len(labels)=}, {len(all_predictions)=}")
            return accuracy(all_predictions, labels)

    def train(
        self,
        training_data: BOWDataset,
        val_data: BOWDataset,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
    ) -> None:
        """Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        torch.manual_seed(0)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
            for inputs_b_l, lengths_b, labels_b in tqdm(dataloader):
              
              # zero the parameter gradients
              optimizer.zero_grad()
      
              # forward + backward + optimize
              outputs = self.model(inputs_b_l, lengths_b)
              loss = criterion(outputs, labels_b)
              per_dp_loss = loss.item()
              total_loss += per_dp_loss
              loss.backward()
              optimizer.step()

            # TODO: Implement this!
            val_acc = self.evaluate(val_data)

            print(
                f"Epoch: {epoch + 1:<2} | Loss: {total_loss / len(dataloader):.4f} | Val accuracy: {100 * val_acc:.2f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    args = parser.parse_args()

    num_epochs = args.epochs
    lr = args.learning_rate
    data_type = DataType(args.data)

    train_data, val_data, dev_data, test_data = load_data(data_type)

    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)

    max_length = 100
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds = BOWDataset(test_data, tokenizer, label2id, max_length)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
    )

    trainer = Trainer(model)

    print("Training the model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer.train(train_ds, val_ds, optimizer, num_epochs)

    # Evaluate on dev
    dev_acc = trainer.evaluate(dev_ds)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on test
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]
    save_results(
        test_data,
        test_preds,
        os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
    )
