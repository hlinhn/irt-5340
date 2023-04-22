#!/usr/bin/env python3


from itertools import islice
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import pickle
import scipy.sparse
import sys


ROOT = '/home/linh/Downloads/ednet-project'
data_folder = f'{ROOT}/KT1'
question_content_path = f'{ROOT}/ednet_contents/questions.csv'
generated_path = f'{ROOT}/ednet_generated'


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


def make_student_id(folder=data_folder):
    all_files = [f for f in os.listdir(folder)]
    indices = [x for x in range(len(all_files))]
    sid = dict([(student, index) for student, index in zip(all_files, indices)])
    with open(f'{generated_path}/student_lookup.pkl', 'wb') as fp:
        pickle.dump(sid, fp)


def make_question_id():
    questions = pd.read_csv(question_content_path)
    indices = [x for x in range(len(questions))]
    qid = dict([(question_id, index) for question_id, index in zip(questions.question_id, indices)])
    with open(f'{generated_path}/question_lookup.pkl', 'wb') as fp:
        pickle.dump(qid, fp)


def generate_compressed_data():
    sid = {}
    qid = {}
    with open(f'{generated_path}/student_lookup.pkl', 'rb') as fp:
        sid = pickle.load(fp)
    with open(f'{generated_path}/question_lookup.pkl', 'rb') as fp:
        qid = pickle.load(fp)

    questions = pd.read_csv(question_content_path)
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


def write_student_assist_format(train, valid, key, qdata, qid, filter_qid=False):
    keyname = f"{key}.csv"
    data = pd.read_csv(os.path.join(ROOT, keyname))
    if filter_qid:
        data = data[data['question_id'].isin(qid.keys())]
    total = len(data)
    train_len = int(total * 0.7)
    np_data = data[["question_id", "user_answer", "elapsed_time"]].to_numpy()
    correct_ans = [qdata[x] for x in np_data[:, 0]]
    correct = (np_data[:, 1] == correct_ans).astype(int).astype(str)
    question_number = np.array([qid[x] for x in np_data[:, 0]]).astype(str)
    np_data = np_data.astype(str)

    train_lines = [str(key), str(train_len),
                   ','.join(question_number[:train_len]),
                   ','.join(correct[:train_len]),
                   ','.join(np_data[:train_len, 2]) + "\n"]
    valid_lines = [str(key), str(total - train_len),
                   ','.join(question_number[train_len:]),
                   ','.join(correct[train_len:]),
                   ','.join(np_data[train_len:, 2]) + "\n"]
    train_str = "\n".join(train_lines)
    valid_str = "\n".join(valid_lines)
    train.write(train_str)
    valid.write(valid_str)


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


def generate_assist_format(filename, folder=generated_path):
    questions = pd.read_csv(question_content_path)
    question_data = dict([(id, ans) for id, ans in zip(questions.question_id, questions.correct_answer)])

    os.makedirs(folder, exist_ok=True)
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
            write_student_assist_format(train_file, valid_file, key, question_data, qid_data, filter_qid=True)
        train_file.close()
        valid_file.close()


if __name__ == '__main__':
    if sys.argv[1] == "interact":
        count_interactions(sys.argv[2])
    if sys.argv[1] == "compressed":
        make_question_id()
        make_student_id()
        generate_data()
    if sys.argv[1] == "assist":
        generate_assist_format(sys.argv[2])
    if sys.argv[1] == "load":
        dataset(sys.argv[2])
