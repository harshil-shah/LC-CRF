import os
import time
import numpy as np
import tensorflow as tf


class RunBase(object):

    def __init__(self, trainer, trainer_kwargs, vocab, labels, neg_label, n_data, data_dims, out_dir):

        self.trainer = trainer(**trainer_kwargs)

        self.vocab = vocab
        self.labels = labels
        self.neg_label = neg_label

        self.n_data = n_data
        self.data_dims = data_dims

        self.checkpoint = tf.train.Checkpoint(optimiser=self.trainer.optimiser,
                                              **{v.name: v for v in self.trainer.trainable_variables})

        self.out_dir = out_dir

    def evaluate(self, dataset, **kwargs):

        objective = 0
        x_all = []
        y_all = []
        y_pred_all = []

        for _, x, y in dataset:

            objective += self.trainer.objective(x, y).numpy() * len(x.numpy())
            y_pred, score = self.trainer.predict(x)

            x_all.append(x.numpy())
            y_all.append(y.numpy())
            y_pred_all.append(y_pred.numpy())

        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)

        objective /= len(y_all)

        prc, rec, f1 = self.f1(x_all, y_all, y_pred_all)

        return objective, prc, rec, f1

    def f1(self, x, y_true, y_pred):

        tp = 0
        fp = 0
        fn = 0

        x = x.flatten()
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        neg = self.labels.index(self.neg_label)

        for n in range(len(x)):
            if self.vocab[x[n]] != '<PAD>':
                if y_pred[n] != neg:
                    if y_true[n] == y_pred[n]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if y_true[n] != y_pred[n]:
                        fn += 1

        prc = tp / (tp + fp)
        rec = tp / (tp + fn)

        f1 = (2 * prc * rec) / (prc + rec)

        return prc, rec, f1

    def acc(self, x, y_true, y_pred):

        p = 0
        a = 0

        x = x.flatten()
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        for n in range(len(x)):
            if self.vocab[x[n]] != '<PAD>':
                a += 1
                if y_true[n] == y_pred[n]:
                    p += 1

        prc = p / a
        rec = p / a

        f1 = p / a

        return prc, rec, f1


class RunBaseNormal(RunBase):

    def __init__(self, trainer, trainer_kwargs, vocab, labels, neg_label, dataset_dir, n_data, data_dims, out_dir):

        super().__init__(trainer, trainer_kwargs, vocab, labels, neg_label, n_data, data_dims, out_dir)

        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = \
            self.load_data(dataset_dir)

    def load_data(self, dataset_dir):

        x_train = np.memmap(os.path.join(dataset_dir, 'text_train.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['train']] + self.data_dims))
        y_train = np.memmap(os.path.join(dataset_dir, 'labels_train.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['train']] + self.data_dims))

        x_valid = np.memmap(os.path.join(dataset_dir, 'text_valid.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['valid']] + self.data_dims))
        y_valid = np.memmap(os.path.join(dataset_dir, 'labels_valid.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['valid']] + self.data_dims))

        x_test = np.memmap(os.path.join(dataset_dir, 'text_test.mmap'), dtype=np.uint16, mode='r',
                           shape=tuple([self.n_data['test']] + self.data_dims))
        y_test = np.memmap(os.path.join(dataset_dir, 'labels_test.mmap'), dtype=np.uint16, mode='r',
                           shape=tuple([self.n_data['test']] + self.data_dims))

        return x_train, y_train, x_valid, y_valid, x_test, y_test


class RunActiveSL(RunBaseNormal):

    def create_dataset(self, x, y, n_batch, avail_inds=None, train=True):

        if avail_inds is None:
            avail_inds = np.arange(len(x))

        def gen():

            if train:
                i = 1

                while True:
                    inds = np.random.choice(avail_inds, size=min((n_batch, len(avail_inds))), replace=False)
                    yield i, x[inds], y[inds]
                    i += 1

            else:
                for i in range(0, len(avail_inds), n_batch):
                    yield i, x[avail_inds[i: i + n_batch]], y[avail_inds[i: i + n_batch]]

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.int32, tf.int32, tf.int32),
            output_shapes=([], [None] + self.data_dims, [None] + self.data_dims)
        )

        return dataset

    def train(self, n_data_init, n_data_per_iteration, n_iterations, n_batch, n_batch_val):

        dataset_valid = self.create_dataset(self.x_valid, self.y_valid, n_batch_val, train=False)

        inds = np.random.choice(len(self.x_train), size=n_data_init, replace=False)

        j = 0

        while len(inds) < len(self.x_train):

            start = time.perf_counter()

            inds_new = self.get_next_inds(n_data_per_iteration, n_batch_val, inds)

            inds = np.append(inds, inds_new, axis=0)

            print('Outer iteration ' + str(j) + ':'
                  + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)'
                  )

            dataset_train = self.create_dataset(self.x_train, self.y_train, n_batch, inds)

            for i, x_train_i, y_train_i in dataset_train:

                start = time.perf_counter()

                objective = self.trainer.optimise(x_train_i, y_train_i)

                print('Iteration ' + str(i.numpy()) + ':'
                      + ' objective = ' + str(objective.numpy())
                      + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)'
                      )

                if i >= n_iterations:
                    break

            objective_val, prc_val, rec_val, f1_val = self.evaluate(dataset_valid)

            print('Validation:'
                  + ' objective = ' + str(objective_val)
                  + ' P = ' + str(prc_val)
                  + ' R = ' + str(rec_val)
                  + ' F1 = ' + str(f1_val)
                  )

            self.checkpoint.write(os.path.join(self.out_dir, 'checkpoint'))

            j += 1

    def get_next_inds(self, n_inds, n_batch, inds_old):

        avail_inds = np.array([i for i in range(len(self.x_train)) if i not in inds_old])

        dataset = self.create_dataset(self.x_train, self.y_train, n_batch, avail_inds, train=False)

        scores_all = []

        for _, x, y in dataset:

            y_pred, score = self.trainer.predict(x)

            scores_all.append(score.numpy())

        scores_all = np.concatenate(scores_all, axis=0)

        inds = np.argsort(scores_all)[:n_inds]

        return avail_inds[inds]


class RunActiveSSL(RunBaseNormal):

    def create_dataset_sup(self, x, y, n_batch, avail_inds=None):

        if avail_inds is None:
            avail_inds = np.arange(len(x))

        def gen():
            for i in range(0, len(avail_inds), n_batch):
                yield i, x[avail_inds[i: i + n_batch]], y[avail_inds[i: i + n_batch]]

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.int32, tf.int32, tf.int32),
            output_shapes=([], [None] + self.data_dims, [None] + self.data_dims)
        )

        return dataset

    def create_dataset_semisup(self, x, y, n_batch_sup, n_batch_unsup, avail_inds_sup):

        avail_inds_unsup = np.array([i for i in range(len(x)) if i not in avail_inds_sup])

        def gen():

            i = 1

            while True:
                inds_sup = np.random.choice(avail_inds_sup, size=min((n_batch_sup, len(avail_inds_sup))), replace=False)
                inds_unsup = np.random.choice(avail_inds_unsup, size=min((n_batch_unsup, len(avail_inds_unsup))),
                                              replace=False)
                yield i, x[inds_sup], y[inds_sup], x[inds_unsup]
                i += 1

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
            output_shapes=([], [None] + self.data_dims, [None] + self.data_dims, [None] + self.data_dims)
        )

        return dataset

    def train(self, n_data_init, n_data_per_iteration, n_iterations, n_batch_sup, n_batch_unsup, stop_unsup_data,
              n_batch_val):

        dataset_valid = self.create_dataset_sup(self.x_valid, self.y_valid, n_batch_val)

        inds = np.random.choice(len(self.x_train), size=n_data_init, replace=False)

        j = 0

        while len(inds) < len(self.x_train):

            start = time.perf_counter()

            inds_new = self.get_next_inds(n_data_per_iteration, n_batch_val, inds)

            inds = np.append(inds, inds_new, axis=0)

            print('Outer iteration ' + str(j) + ':'
                  + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)'
                  )

            dataset_train = self.create_dataset_semisup(self.x_train, self.y_train, n_batch_sup, n_batch_unsup, inds)

            for i, x_train_sup, y_train_sup, x_train_unsup in dataset_train:

                start = time.perf_counter()

                if len(self.x_train) - len(inds) < stop_unsup_data:
                    objective = self.trainer.optimise_sl(x_train_sup, y_train_sup)

                    print('Iteration ' + str(i.numpy()) + ':'
                          + ' objective = ' + str(objective.numpy())
                          + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)'
                          )

                else:
                    objective, objective_sup, objective_unsup = self.trainer.optimise(x_train_sup, y_train_sup,
                                                                                      x_train_unsup)

                    print('Iteration ' + str(i.numpy()) + ':'
                          + ' objective = ' + str(objective.numpy())
                          + ' objective_sup = ' + str(objective_sup.numpy())
                          + ' objective_unsup = ' + str(objective_unsup.numpy())
                          + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)'
                          )

                if i >= n_iterations:
                    break

            objective_val, prc_val, rec_val, f1_val = self.evaluate(dataset_valid)

            print('Validation:'
                  + ' objective = ' + str(objective_val)
                  + ' P = ' + str(prc_val)
                  + ' R = ' + str(rec_val)
                  + ' F1 = ' + str(f1_val)
                  )

            self.checkpoint.write(os.path.join(self.out_dir, 'checkpoint'))

            j += 1

    def get_next_inds(self, n_inds, n_batch, inds_old):

        avail_inds = np.array([i for i in range(len(self.x_train)) if i not in inds_old])

        dataset = self.create_dataset_sup(self.x_train, self.y_train, n_batch, avail_inds)

        scores_all = []

        for _, x, y in dataset:

            y_pred, score = self.trainer.predict(x)

            scores_all.append(score.numpy())

        scores_all = np.concatenate(scores_all, axis=0)

        inds = np.argsort(scores_all)[:n_inds]

        return avail_inds[inds]


class RunActiveSSLExtraData(RunBase):

    def __init__(self, trainer, trainer_kwargs, vocab, labels, neg_label, dataset_dir, n_data, data_dims, out_dir):

        super().__init__(trainer, trainer_kwargs, vocab, labels, neg_label, n_data, data_dims, out_dir)

        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test, self.x_unsup = \
            self.load_data(dataset_dir)

    def load_data(self, dataset_dir):

        x_train = np.memmap(os.path.join(dataset_dir, 'text_train.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['train']] + self.data_dims))
        y_train = np.memmap(os.path.join(dataset_dir, 'labels_train.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['train']] + self.data_dims))

        x_valid = np.memmap(os.path.join(dataset_dir, 'text_valid.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['valid']] + self.data_dims))
        y_valid = np.memmap(os.path.join(dataset_dir, 'labels_valid.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['valid']] + self.data_dims))

        x_test = np.memmap(os.path.join(dataset_dir, 'text_test.mmap'), dtype=np.uint16, mode='r',
                           shape=tuple([self.n_data['test']] + self.data_dims))
        y_test = np.memmap(os.path.join(dataset_dir, 'labels_test.mmap'), dtype=np.uint16, mode='r',
                           shape=tuple([self.n_data['test']] + self.data_dims))

        x_unsup = np.memmap(os.path.join(dataset_dir, 'text_unsup.mmap'), dtype=np.uint16, mode='r',
                            shape=tuple([self.n_data['unsup']] + self.data_dims))

        return x_train, y_train, x_valid, y_valid, x_test, y_test, x_unsup

    def create_dataset_sup(self, x, y, n_batch, avail_inds=None):

        if avail_inds is None:
            avail_inds = np.arange(len(x))

        def gen():
            for i in range(0, len(avail_inds), n_batch):
                yield i, np.int32(x[avail_inds[i: i + n_batch]]), np.int32(y[avail_inds[i: i + n_batch]])

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.int32, tf.int32, tf.int32),
            output_shapes=([], [None] + self.data_dims, [None] + self.data_dims)
        )

        return dataset

    def create_dataset_semisup(self, x, y, n_batch_sup, n_batch_unsup, avail_inds_sup):

        avail_inds_unsup = np.arange(len(self.x_unsup))

        def gen():

            i = 1

            while True:
                inds_sup = np.random.choice(avail_inds_sup, size=min((n_batch_sup, len(avail_inds_sup))), replace=False)
                inds_unsup = np.random.choice(avail_inds_unsup, size=min((n_batch_unsup, len(avail_inds_unsup))),
                                              replace=False)
                yield i, np.int32(x[inds_sup]), np.int32(y[inds_sup]), np.int32(self.x_unsup[inds_unsup])
                i += 1

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
            output_shapes=([], [None] + self.data_dims, [None] + self.data_dims, [None] + self.data_dims)
        )

        return dataset

    def train(self, n_data_init, n_data_per_iteration, n_iterations, n_batch_sup, n_batch_unsup, n_batch_val):

        dataset_valid = self.create_dataset_sup(self.x_valid, self.y_valid, n_batch_val)

        inds = np.random.choice(len(self.x_train), size=n_data_init, replace=False)

        j = 0

        while len(inds) < len(self.x_train):

            start = time.perf_counter()

            inds_new = self.get_next_inds(n_data_per_iteration, n_batch_val, inds)

            inds = np.append(inds, inds_new, axis=0)

            print('Outer iteration ' + str(j) + ':'
                  + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)'
                  )

            dataset_train = self.create_dataset_semisup(self.x_train, self.y_train, n_batch_sup, n_batch_unsup, inds)

            for i, x_train_sup, y_train_sup, x_train_unsup in dataset_train():

                start = time.perf_counter()

                objective, objective_sup, objective_unsup = self.trainer.optimise(x_train_sup, y_train_sup,
                                                                                  x_train_unsup)

                print('Iteration ' + str(i.numpy()) + ':'
                      + ' objective = ' + str(objective.numpy())
                      + ' objective_sup = ' + str(objective_sup.numpy())
                      + ' objective_unsup = ' + str(objective_unsup.numpy())
                      + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)'
                      )

                if i >= n_iterations:
                    break

            objective_val, prc_val, rec_val, f1_val = self.evaluate(dataset_valid)

            print('Validation:'
                  + ' objective = ' + str(objective_val)
                  + ' P = ' + str(prc_val)
                  + ' R = ' + str(rec_val)
                  + ' F1 = ' + str(f1_val)
                  )

            self.checkpoint.write(os.path.join(self.out_dir, 'checkpoint'))

            j += 1

    def get_next_inds(self, n_inds, n_batch, inds_old):

        avail_inds = np.array([i for i in range(len(self.x_train)) if i not in inds_old])

        dataset = self.create_dataset_sup(self.x_train, self.y_train, n_batch, avail_inds)

        scores_all = []

        for _, x, y in dataset():

            y_pred, score = self.trainer.predict(x)

            scores_all.append(score.numpy())

        scores_all = np.concatenate(scores_all, axis=0)

        inds = np.argsort(scores_all)[:n_inds]

        return avail_inds[inds]
