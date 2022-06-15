import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math

from dataloader import *
from neuralnet import *

if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    data = DataLoader() # data loader does everything for u

    train_losses, val_losses, train_accs, val_accs = train(model, data.X_train, data.y_train, data.X_val, data.y_val, config)

    test_acc = test(model, data.X_test, data.y_test)

    # TODO: Plots
    print("Accuracy: ", test_acc)
    plot(train_losses, val_losses, 'Loss')
    plot(train_accs, val_accs, 'Loss')