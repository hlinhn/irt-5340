#!/usr/bin/env python3


import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import scipy.sparse
import pickle
from itertools import islice
import torch


ROOT = '/home/linh/Downloads/KT1'

data_folder = "/home/linh/Downloads/ednet-project/KT1"
ROOT_FOLDER = '/home/linh/projects/deep-irt'


class EdnetKT1(torch.utils.data.Dataset):
    def __init__(self, ds_file='kt1_data_0.npz', train=True, binarized_response=True, swap_ability_item=False, **kwargs):
        super().__init__()

        self.swap_ability_item = swap_ability_item

        raw_data = scipy.sparse.load_npz(os.path.join(ROOT_FOLDER, ds_file)).todok()

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
        self.maskperc = 0.3

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
        return missing_data, response, missing_mask, mask


def test_dataset_loader():
    test_loader = EdnetKT1(train=True)
    loader = torch.utils.data.DataLoader(test_loader, batch_size=32, shuffle=True)
    for batch_idx, (md, r, mi, m) in enumerate(loader):
        print(md)
        print(r.shape)
        break


def chunks(data, SIZE=100000):
   it = iter(data)
   for i in range(0, len(data), SIZE):
      yield {k for k in islice(it, SIZE)}


def process_student(student_id, qdata, qid, sid, sparse_mat):
    data = pd.read_csv(os.path.join(data_folder, student_id))
    np_data = data[["question_id", "user_answer", "elapsed_time"]].to_numpy()
    correct_ans = [qdata[x] for x in np_data[:, 0]]
    correct = (np_data[:, 1] == correct_ans)
    incorrect = np.invert(correct)
    question_number = np.array([qid[x] for x in np_data[:, 0]])
    correct_qid = question_number[np.nonzero(correct)]
    incorrect_qid = question_number[np.nonzero(incorrect)]
    sparse_mat[sid, correct_qid] = 2
    sparse_mat[sid, incorrect_qid] = 1
    return sparse_mat


def make_student_id(folder):
    all_files = [f for f in os.listdir(folder)]
    indices = [x for x in range(len(all_files))]
    sid = dict([(student, index) for student, index in zip(all_files, indices)])
    with open('student_lookup.pkl', 'wb') as fp:
        pickle.dump(sid, fp)


def make_question_id():
    questions = pd.read_csv('/home/linh/Downloads/ednet-project/ednet_contents/questions.csv')
    indices = [x for x in range(len(questions))]
    qid = dict([(question_id, index) for question_id, index in zip(questions.question_id, indices)])
    with open('question_lookup.pkl', 'wb') as fp:
        pickle.dump(qid, fp)


def generate_data():
    sid = {}
    qid = {}
    with open('student_lookup.pkl', 'rb') as fp:
        sid = pickle.load(fp)
    with open('question_lookup.pkl', 'rb') as fp:
        qid = pickle.load(fp)

    questions = pd.read_csv('/home/linh/Downloads/ednet-project/ednet_contents/questions.csv')
    question_data = dict([(id, ans) for id, ans in zip(questions.question_id, questions.correct_answer)])

    counter = 0
    for item in chunks(sid.keys()):
        sparse = scipy.sparse.dok_array((len(sid), len(qid)), dtype=np.int8)
        for key in tqdm(item):
            sparse = process_student(key, question_data, qid, sid[key], sparse)

        scipy.sparse.save_npz(f'kt1_data_{counter}.npz', sparse.tocoo())
        counter += 1

def load_dataset(filename):
    mat = scipy.sparse.load_npz(filename).todok()
    row = mat[0, :].toarray()
    print(row.shape)
    print(len(mat))


def write_student(train, valid, key, qdata, qid, filter_qid=False):
    keyname = f"{key}.csv"
    data = pd.read_csv(os.path.join(ROOT, keyname))
    if filter_qid:
        data = data[data['question_id'].isin(qid.keys())]
    total = len(data)
    # train_len = int(total * 0.7)
    np_data = data[["question_id", "user_answer", "elapsed_time"]].to_numpy()
    correct_ans = [qdata[x] for x in np_data[:, 0]]
    correct = (np_data[:, 1] == correct_ans).astype(int).astype(str)
    question_number = np.array([qid[x] for x in np_data[:, 0]]).astype(str)
    np_data = np_data.astype(str)
    train_lines = [str(key), str(train_len), ','.join(question_number), ','.join(correct) + "\n"], #, ','.join(np_data[:train_len, 2]) + "\n"]
    # valid_lines = [str(key), str(total - train_len), ','.join(question_number[train_len:]), ','.join(correct[train_len:]), ','.join(np_data[train_len:, 2]) + "\n"]
    train_str = "\n".join(train_lines)
    # valid_str = "\n".join(valid_lines)
    train.write(train_str)
    # valid.write(valid_str)


def count_interaction_student(key, qid, filter_qid=True):
    keyname = f"{key}.csv"
    data = pd.read_csv(os.path.join(data_folder, keyname))
    if filter_qid:
        data = data[data['question_id'].isin(qid.keys())]
    total = len(data)
    return total


def count_interactions(filename):
    with open(filename, 'rb') as f:
        data = np.load(f)
        qid_file = open(f"{filename[:-4]}_qid.npy", 'rb')
        qid_data_raw = np.load(qid_file)
        print(qid_data_raw.shape)
        qid_file.close()
        qid_data = dict([(question, num + 1) for num, question in enumerate(qid_data_raw)])
        acc = 0
        for k in tqdm(data):
            number = count_interaction_student(k, qid_data)
            acc += number
    print(acc)


def run_test(filename, folder="/home/linh/Downloads/ednet_generated"):
    questions = pd.read_csv('/home/linh/Downloads/ednet-project/ednet_contents/questions.csv')
    question_data = dict([(id, ans) for id, ans in zip(questions.question_id, questions.correct_answer)])

    with open(filename, 'rb') as f:
        data = np.load(f)
        basename = os.path.basename(filename)
        qid_file = open(f"{filename[:-4]}_qid.npy", 'rb')
        qid_data_raw = np.load(qid_file)
        print(qid_data_raw.shape)
        qid_file.close()
        qid_data = dict([(question, num + 1) for num, question in enumerate(qid_data_raw)])
        train_file = open(os.path.join(folder, f"{basename[:-4]}_seq_train.csv"), 'w')
        valid_file = open(os.path.join(folder, f"{basename[:-4]}_seq_valid.csv"), 'w')
        for key in tqdm(data):
            write_student(train_file, valid_file, key, question_data, qid_data, filter_qid=True)
        train_file.close()
        valid_file.close()


if __name__ == '__main__':
    count_interactions(sys.argv[1])
    # test_dataset_loader()
    # load_dataset(sys.argv[1])
    # generate_data()
    # make_question_id()
    # make_student_id(sys.argv[1])
    # run_test(sys.argv[1])
