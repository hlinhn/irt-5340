#!/usr/bin/env python3

import os
import torch
import shutil
from tqdm import tqdm
from .config import *


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def get_annealing_factor(epoch, which_mini_batch, N_mini_batches):
    if anneal_kl:
        annealing_factor = \
            (float(which_mini_batch + epoch * N_mini_batches + 1) /
             float(epochs // 2 * N_mini_batches))
    else:
        annealing_factor = beta_kl
    return annealing_factor


def train(model, optimizer, train_loader, epoch):
    model.train()
    train_loss = AverageMeter()
    pbar = tqdm(total=len(train_loader), position=0, leave=True)

    for batch_idx, (_, response, _, mask) in enumerate(train_loader):
        mb = response.size(0)
        response = response.to(device)
        mask = mask.long().to(device)
        annealing_factor = get_annealing_factor(epoch, batch_idx, len(train_loader))

        optimizer.zero_grad()

        outputs = model(response, mask)
        loss = model.elbo(*outputs, annealing_factor=annealing_factor, use_kl_divergence=True)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), mb)

        pbar.update()
        pbar.set_postfix({'Loss': train_loss.avg})

    pbar.close()
    print('====> Train Epoch: {} Loss: {:.4f}'.format(epoch, train_loss.avg))

    return train_loss.avg


def test(model, test_loader, epoch):
    model.eval()
    test_loss = AverageMeter()
    pbar = tqdm(total=len(test_loader))

    with torch.no_grad():
        for _, response, _, mask in test_loader:
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)

            outputs = model(response, mask)
            loss = model.elbo(*outputs)
            test_loss.update(loss.item(), mb)

            pbar.update()
            pbar.set_postfix({'Loss': test_loss.avg})

    pbar.close()
    print('====> Test Epoch: {} Loss: {:.4f}'.format(epoch, test_loss.avg))

    return test_loss.avg


def get_log_marginal_density(model, loader):
    model.eval()
    meter = AverageMeter()
    pbar = tqdm(total=len(loader))

    with torch.no_grad():
        for _, response, _, mask in loader:
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)

            marginal = model.log_marginal(
                response,
                mask,
                num_samples=num_posterior_samples,
            )
            marginal = torch.mean(marginal)
            meter.update(marginal.item(), mb)

            pbar.update()
            pbar.set_postfix({'Marginal': meter.avg})

    pbar.close()
    print('====> Marginal: {:.4f}'.format(meter.avg))

    return meter.avg
