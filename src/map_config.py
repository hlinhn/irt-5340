#!/usr/bin/env python3


irt_model = '1pl'           # Available values: '1pl', '2pl' and '3pl'
dataset = 'ednet-kt1'
ability_dim = 16
# From Trieu: Optional suggestion
# You can use deep-learning network to implement the decoding phase.
generative_model = 'irt'    # Available values: 'irt', 'link', 'deep', 'residual'

artificial_missing_perc = 0.3

num_person = len(raw_data['user_id'].unique()) * 0.7
num_item = len(raw_data['question_id'].unique())

lr = 5e-3
batch_size = 32
epochs = 5
max_iters = -1
num_workers = 0

cuda = True
gpu_device = 0
