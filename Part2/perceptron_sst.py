"""Standalone perceptron script for the sst2 dataset."""

import os
from typing import Set

from features import make_featurize
from perceptron import PerceptronModel, featurize_data
from utils import DataType, load_data

# Hardcoded hyperparameters
DATA_TYPE = DataType.SST2
FEATURE_TYPES: Set[str] = {"bow", "len", "exc_c", "pron_c", "log_wc", "pos_c", "neg_c"}
NUM_EPOCHS = 30
LR = 0.1

if __name__ == "__main__":
    train_data, val_data, dev_data, test_data = load_data(DATA_TYPE)

    train_data = featurize_data(train_data, FEATURE_TYPES)
    val_data = featurize_data(val_data, FEATURE_TYPES)
    dev_data = featurize_data(dev_data, FEATURE_TYPES)
    test_data = featurize_data(test_data, FEATURE_TYPES)

    model = PerceptronModel()
    print("Training the model...")
    model.train(train_data, val_data, NUM_EPOCHS, LR)

    # Predict on the development set.
    dev_acc = model.evaluate(
        dev_data,
        save_path=os.path.join(
            "results",
            "perceptron_sst2_dev_predictions.csv",
        ),
    )
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on the test set
    _ = model.evaluate(
        test_data,
        save_path=os.path.join(
            "results",
            "perceptron_sst2_test_predictions.csv",
        ),
    )

    model.save_weights(
        os.path.join("results", "perceptron_sst2_model.json")
    )
