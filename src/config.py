#!/usr/bin/env python3


irt_model = '2pl'           # Available values: '1pl', '2pl' and '3pl'
dataset = 'ednet-kt1'
ability_dim = 8
ability_merge = 'product'   # Available values: 'product' and 'mean'
conditional_posterior = True
generative_model = 'irt'    # Available values: 'irt', 'link', 'deep', 'residual'
response_dist = 'gaussian' # Available values: 'bernoulli' and 'gaussian'
drop_missing = False
artificial_missing_perc = 0.3
no_test = False
num_person = 70000
num_item = 13169
# num_person = len(raw_data['user_id'].unique()) * 0.7
# num_item = len(raw_data['question_id'].unique())
num_posterior_samples = 10
hidden_dim = 64
max_num_person = None
max_num_item = None
swap_ability_item = False

lr = 5e-3
batch_size = 32
epochs = 5
max_iters = -1
num_workers = 0
anneal_kl = False
beta_kl = 0.9

seed = 8
cuda = True
gpu_device = 0

torch.manual_seed(seed)
np.random.seed(seed)

num_person = None
num_item = None

if max_num_person is not None:
    max_num_person = int(max_num_person)

if max_num_item is not None:
    max_num_item = int(max_num_item)
