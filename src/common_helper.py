#!/usr/bin/env python3


from .config import *
from .model import VIBO_1PL, VIBO_2PL, VIBO_3PL
from .datasets import EdnetKT1Compressed

import os
import torch


def make_outdir():
    out_file = 'VIBO_{}_{}_{}_{}_{}maskperc_{}ability_{}_{}_seed{}_conditional_posterior{}_batch_size{}_epochs{}_lr{}_anneal_kl{}_beta_kl{}'.format(
        irt_model,
        dataset,
        response_dist,
        generative_model,
        artificial_missing_perc,
        ability_dim,
        ability_merge,
        'conditional_q' if conditional_posterior else 'unconditional_q',
        seed,
        conditional_posterior,
        batch_size,
        epochs,
        lr,
        anneal_kl,
        beta_kl,
    )
    out_dir = os.path.join(OUT_DIR, out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return out_dir


def make_model():
    device = torch.device('cuda' if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(gpu_device)
    if irt_model == '1pl':
        model_class = VIBO_1PL
    elif irt_model == '2pl':
        model_class = VIBO_2PL
    elif irt_model == '3pl':
        model_class = VIBO_3PL
    else:
        raise Exception(f'model {irt_model} not recognized')

    model = model_class(
        ability_dim,
        num_item,
        hidden_dim = hidden_dim,
        ability_merge = ability_merge,
        conditional_posterior = conditional_posterior,
        generative_model = generative_model,
        response_dist = response_dist,
        replace_missing_with_prior = not drop_missing,
    ).to(device)


def make_dataloaders():
    train_dataset = EdnetKT1Compressed(train=True, maskperc=0.0, binarized_response=False, swap_ability_item=swap_ability_item)
    test_dataset  = EdnetKT1Compressed(train=False, maskperc=artificial_missing_perc, binarized_response=False, swap_ability_item=swap_ability_item)

    num_person = train_dataset.num_person
    num_item = train_dataset.num_item

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )
    return train_loader, test_loader
