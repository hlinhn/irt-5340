#!/usr/bin/env python3

from .training_helper import *
from .config import *
from .common_paths import *
from .common_helper import make_outdir, make_model, make_dataloaders

import torch
import os
import math
from torch import optim
from time import time
import numpy as np


def main():
    out_dir = make_outdir()

    if max_iters != -1:
        epochs = int(math.ceil(max_iters / float(len(train_loader))))
        print(f'Found MAX_ITERS={max_iters}, setting EPOCHS={epochs}')

    train_loader, test_loader = make_dataloaders()
    model = make_model()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    is_best, best_loss = False, np.inf
    train_losses = np.zeros(epochs)
    if not no_test:
        test_losses  = np.zeros(epochs)

    train_times = np.zeros(epochs)

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, optimizer, train_loader, epoch)
        end_time = time.time()
        train_losses[epoch] = train_loss
        train_times[epoch] = start_time - end_time

        if not no_test:
            test_loss = test(model, test_loader, epoch)
            test_losses[epoch] = test_loss
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
        else:
            is_best = train_loss < best_loss
            best_loss = min(train_loss, best_loss)

        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
        }, is_best, folder=out_dir)

        np.save(os.path.join(out_dir, 'train_losses.npy'), train_losses)
        np.save(os.path.join(out_dir, 'train_times.npy'), train_times)

        if not no_test:
            np.save(os.path.join(out_dir, 'test_losses.npy'),  test_losses)

    print(f'Train time: {np.abs(train_times).sum()}')


if __name__ == '__main__':
    main()
