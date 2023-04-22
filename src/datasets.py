#!/usr/bin/env python3

from .common_paths import *

import torch
import os
import pandas as pd
import numpy as np
import scipy.sparse
import copy


class EdnetKT1(torch.utils.data.Dataset):
    def __init__(self, ds_file='kt1_subset.csv', train=True, binarized_response=True, swap_ability_item=False, **kwargs):
        super().__init__()

        self.swap_ability_item = swap_ability_item

        name, _ = os.path.splitext(ds_file)
        raw_data = pd.read_csv(os.path.join(EDNET_KT1_DIR, ds_file))

        responses = self.make_score_matrix(raw_data, name=name)

        rs = np.random.RandomState(42)
        swapper = np.arange(responses.shape[0])
        rs.shuffle(swapper)
        responses = responses[swapper]

        self.binarized_response = binarized_response
        if binarized_response:
            responses = np.round(responses)

        num_users = responses.shape[0]
        num_train = int(0.7 * num_users)

        if train:
            responses = responses[:num_train]
        else:
            responses = responses[num_train:]

        response_mask = np.ones_like(responses)
        response_mask[responses == MISSING_DATA] = 0

        self.responses = responses
        self.mask = response_mask
        self.length = responses.shape[0]
        self.num_person = responses.shape[0]
        self.num_item = responses.shape[1]

    def make_score_matrix(self, raw_data, name):
        cache_file = os.path.join(EDNET_KT1_DIR, f'{name}_score_matrix_swap{self.swap_ability_item}.npy'.lower())
        if os.path.isfile(cache_file):
            return np.load(cache_file)

        raw_data['response'] = 0
        raw_data.loc[raw_data['user_answer'] == raw_data['correct_answer'], 'response'] = 1

        compact_score_mat = raw_data.groupby(['user_id', 'question_id'])['response'].mean().reset_index()
        score_mat = compact_score_mat.pivot(index='user_id', columns='question_id', values='response')
        score_mat = score_mat.fillna(MISSING_DATA)
        score_mat = score_mat.values

        if self.swap_ability_item:
            score_mat = np.transpose(score_mat)

        np.save(cache_file, score_mat)
        return score_mat

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        response = self.responses[index]
        item_id = np.arange(self.num_item)
        item_id[response == MISSING_DATA] = MISSING_DATA
        mask = self.mask[index]

        response = torch.from_numpy(response).float().unsqueeze(1)
        item_id = torch.from_numpy(item_id).long().unsqueeze(1)
        mask = torch.from_numpy(mask).bool().unsqueeze(1)

        return index, response, item_id, mask


class EdnetKT1Compressed(torch.utils.data.Dataset):
    def __init__(self, ds_file='kt1_data_0.npz', train=True, maskperc=0.3, binarized_response=True, swap_ability_item=False, **kwargs):
        super().__init__()

        self.swap_ability_item = swap_ability_item

        raw_data = scipy.sparse.load_npz(os.path.join(EDNET_KT1_DIR, ds_file)).todok()

        rs = np.random.RandomState(42)
        swapper = np.arange(100000)
        rs.shuffle(swapper)

        num_users = 100000 # responses.shape[0]
        num_train = int(0.7 * num_users)

        if train:
            index_map = swapper[:num_train]
        else:
            index_map = swapper[num_train:]

        self.index_map = index_map
        self.responses = raw_data
        self.length = len(self.index_map) #raw_data.shape[0]
        self.num_person = len(self.index_map) # raw_data.shape[0]
        self.num_item = raw_data.shape[1]
        self.maskperc = maskperc

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        full = self.responses[self.index_map[index], :]
        full_size = full.getnnz()
        unmask_size = int(full_size * (1-self.maskperc))
        rs = np.random.RandomState(42)
        swapper = np.arange(full_size)
        rs.shuffle(swapper)
        masked = swapper[unmask_size:]
        data_indices = self.responses[self.index_map[index], :].nonzero()
        not_keep = (data_indices[0][masked], data_indices[1][masked])
        not_keep_indices = [x for x in zip(*not_keep)]
        response = self.responses[self.index_map[index], :].toarray()
        missing_mask = np.zeros(response.shape, dtype=bool)
        missing_mask[not_keep] = 1
        missing_data = missing_mask * response - 1
        response = response - 1
        response[not_keep] = -1
        mask = np.zeros(response.shape, dtype=bool)
        mask[data_indices] = 1
        mask[not_keep] = 0
        return missing_data.transpose(), response.transpose(), missing_mask.transpose(), mask.transpose()


def artificially_mask_dataset(old_dataset, perc):
    dataset = copy.deepcopy(old_dataset)
    assert perc >= 0 and perc <= 1
    response = dataset.responses
    mask = dataset.mask

    if np.ndim(mask) == 2:
        row, col = np.where(mask != 0)
    elif np.ndim(mask) == 3:
        row, col = np.where(mask[:, :, 0] != 0)
    pool = np.array(list(zip(row, col)))
    num_all = pool.shape[0]
    num = int(perc * num_all)

    rs = np.random.RandomState(42)
    indices = np.sort(
        rs.choice(np.arange(num_all), size=num, replace=False),
    )
    label_indices = pool[indices]
    labels = []
    for idx in label_indices:
        label = copy.deepcopy(response[idx[0], idx[1]])
        labels.append(label)
        mask[idx[0], idx[1]] = 0
        response[idx[0], idx[1]] = -1
    labels = np.array(labels)

    dataset.response = response
    dataset.mask = mask
    dataset.missing_labels = labels
    dataset.missing_indices = label_indices

    return dataset
