from models import LCCRF as Model
from trainers import MaximumLikelihood as Trainer
from run import RunActiveSSL as Run

import os
import sys
import json
import numpy as np
import tensorflow as tf


data_dir = sys.argv[1]
out_dir = sys.argv[2]


dataset_dir = 'CoNLL2003'
dataset_dir = os.path.join(data_dir, dataset_dir)

max_steps = 99999
max_word_len = 99999

n_data = {
    'train': 99999,
    'valid': 99999,
    'test': 99999,
}

data_dims = [max_steps]


with open(os.path.join(dataset_dir, 'tags.json'), 'r') as f:
    labels = json.loads(f.read())

neg_label = 'O'


with open(os.path.join(dataset_dir, 'vocab.json'), 'r') as f:
    vocab = json.loads(f.read())

emb_matrix = np.float32(np.load(os.path.join(dataset_dir, 'emb_matrix.npy')))
char_matrix = np.float32(np.load(os.path.join(dataset_dir, 'char_matrix.npy')))

x_dim = 300
c_dim = 100
y_dim = len(labels)

h_c_dim = 50

nn_phi_kwargs = {
    'depth': 2,
    'units': 600,
    'activation': 'relu',
    'skip': True,
}

nn_eta_kwargs = {
    'depth': 2,
    'units': 600,
    'activation': 'relu',
    'skip': True,
}

nn_xi_kwargs = {
    'depth': 2,
    'units': 600,
    'activation': 'relu',
    'skip': True,
}


model_kwargs = {
    'max_steps': max_steps,
    'max_word_len': max_word_len,
    'x_dim': x_dim,
    'c_dim': c_dim,
    'y_dim': y_dim,
    'h_c_dim': h_c_dim,
    'nn_phi_kwargs': nn_phi_kwargs,
    'nn_eta_kwargs': nn_eta_kwargs,
    'nn_xi_kwargs': nn_xi_kwargs,
}

optimiser = tf.keras.optimizers.SGD

optimiser_kwargs = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'nesterov': True,
}

trainer_kwargs = {'optimiser': optimiser,
                  'optimiser_kwargs': optimiser_kwargs,
                  'model': Model,
                  'model_kwargs': model_kwargs,
                  'emb_matrix': emb_matrix,
                  'char_matrix': char_matrix,
                  'optimise_embs': True,
                  'ssl': True,
                  'ssl_weight_unsup': 0.1,
                  }

train = True

train_n_data_init = 50
train_n_data_per_iteration = 50
train_n_iterations = 50
train_n_batch_sup = 128
train_n_batch_unsup = 128
train_stop_unsup_data = 500
train_n_batch_val = 256


if __name__ == '__main__':

    run = Run(trainer=Trainer,
              trainer_kwargs=trainer_kwargs,
              vocab=vocab,
              labels=labels,
              neg_label=neg_label,
              dataset_dir=dataset_dir,
              n_data=n_data,
              data_dims=data_dims,
              out_dir=out_dir,
              )

    if train:
        run.train(n_data_init=train_n_data_init,
                  n_iterations=train_n_iterations,
                  n_data_per_iteration=train_n_data_per_iteration,
                  n_batch_sup=train_n_batch_sup,
                  n_batch_unsup=train_n_batch_unsup,
                  stop_unsup_data=train_stop_unsup_data,
                  n_batch_val=train_n_batch_val,
                  )
