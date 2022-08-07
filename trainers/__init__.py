import tensorflow as tf


class MaximumLikelihood(object):

    def __init__(self, optimiser, optimiser_kwargs, model, model_kwargs, emb_matrix, char_matrix, optimise_embs=False,
                 ssl=False, ssl_weight_unsup=False):

        self.optimiser = optimiser(**optimiser_kwargs)

        self.emb_matrix = tf.Variable(emb_matrix, name='emb_matrix')
        self.char_matrix = tf.constant(char_matrix)

        self.y_dim = model_kwargs['y_dim']

        self.model = model(**model_kwargs)

        self.trainable_variables = self.model.trainable_variables

        if optimise_embs:
            self.trainable_variables += [self.emb_matrix]

        self.ssl = ssl

        if self.ssl:
            self.optimise = self.optimise_ssl

            if ssl_weight_unsup is None:
                self.ssl_weight_unsup = 1.
            elif ssl_weight_unsup == 'auto':
                self.ssl_weight_unsup = (tf.math.log(tf.cast(self.y_dim, tf.float32)) / tf.math.log(
                    tf.cast(tf.shape(self.emb_matrix)[0], tf.float32)))
            else:
                self.ssl_weight_unsup = tf.constant(ssl_weight_unsup)

        else:
            self.optimise = self.optimise_sl

    def compute_objective_sup(self, x, y):

        x_emb = tf.gather(self.emb_matrix, x)  # (N, T, X)
        x_emb_char = tf.gather(self.char_matrix, x)  # (N, T, L, C)

        y_emb = tf.one_hot(y, depth=self.y_dim)  # (N, T, Y)

        mask = tf.cast(tf.not_equal(x, 0), tf.float32)  # (N, T)
        lens_char = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(x_emb_char, axis=3), 0), tf.int32), axis=2)  # (N, T)

        log_p_y_given_x = self.model.log_p_y_given_x(y_emb, x_emb, x_emb_char, mask, lens_char)  # (N)

        return tf.reduce_mean(log_p_y_given_x)

    def compute_objective_unsup(self, x):

        mask = tf.cast(tf.not_equal(x, 0), tf.float32)  # (N, T)

        lens_char_matrix = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(self.char_matrix, axis=2), 0), tf.int32),
                                         axis=1)  # (V)

        log_p_x = self.model.log_p_x(x, mask, self.emb_matrix, self.char_matrix, lens_char_matrix)  # (N)

        return tf.reduce_mean(log_p_x)

    @tf.function
    def objective(self, x, y):

        objective = self.compute_objective_sup(x, y)

        return objective

    @tf.function
    def optimise_sl(self, x, y):

        with tf.GradientTape() as tape:
            objective = self.compute_objective_sup(x, y)
            loss = -objective

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))

        return objective

    @tf.function
    def optimise_ssl(self, x_sup, y_sup, x_unsup):

        with tf.GradientTape() as tape:
            objective_sup = self.compute_objective_sup(x_sup, y_sup)
            objective_unsup = self.compute_objective_unsup(x_unsup)
            objective = objective_sup + (self.ssl_weight_unsup * objective_unsup)
            loss = -objective

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))

        return objective, objective_sup, objective_unsup

    @tf.function
    def predict(self, x):

        x_emb = tf.gather(self.emb_matrix, x)  # (N, T, X)
        x_emb_char = tf.gather(self.char_matrix, x)  # (N, T, L, C)

        mask = tf.cast(tf.not_equal(x, 0), tf.float32)  # (N, T)
        lens_char = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(x_emb_char, axis=3), 0), tf.int32), axis=2)  # (N, T)

        y, score = self.model.viterbi(x_emb, x_emb_char, mask, lens_char)  # (N, T), (N)

        return y, score
