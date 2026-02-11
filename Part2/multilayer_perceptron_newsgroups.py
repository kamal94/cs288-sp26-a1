"""Standalone MLP script for the newsgroups dataset."""

import os

import torch
from multilayer_perceptron import (
    BOWDataset,
    MultilayerPerceptronModel,
    Tokenizer,
    Trainer,
    get_label_mappings,
)
from utils import DataType, load_data, save_results

# Hardcoded hyperparameters
DATA_TYPE = DataType.NEWSGROUPS
NUM_EPOCHS = 10
LR = 0.0005
MAX_LENGTH = 300

if __name__ == "__main__":
    train_data, val_data, dev_data, test_data = load_data(DATA_TYPE)

    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)

    train_ds = BOWDataset(train_data, tokenizer, label2id, MAX_LENGTH)
    val_ds = BOWDataset(val_data, tokenizer, label2id, MAX_LENGTH)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, MAX_LENGTH)
    test_ds = BOWDataset(test_data, tokenizer, label2id, MAX_LENGTH)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
    )

    trainer = Trainer(model)

    print("Training the model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    trainer.train(train_ds, val_ds, optimizer, NUM_EPOCHS)

    # Evaluate on dev
    dev_acc = trainer.evaluate(dev_ds)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on test
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]
    save_results(
        test_data,
        test_preds,
        os.path.join("results", "mlp_newsgroups_test_predictions.csv"),
    )
