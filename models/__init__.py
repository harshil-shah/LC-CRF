import tensorflow as tf


class LCCRF(object):

    def __init__(self, max_steps, max_word_len, x_dim, c_dim, y_dim, h_c_dim, nn_phi_kwargs, nn_eta_kwargs,
                 nn_xi_kwargs):

        self.max_steps = max_steps
        self.max_word_len = max_word_len

        self.x_dim = x_dim
        self.c_dim = c_dim
        self.y_dim = y_dim

        self.h_c_dim = h_c_dim

        self.nn_c = self.nn_c_fn()
        self.nn_phi = self.nn_x_fn(**nn_phi_kwargs)
        self.nn_eta = self.nn_x_fn(**nn_eta_kwargs)
        self.nn_xi = self.nn_x_fn(**nn_xi_kwargs)

        self.A0 = tf.Variable(tf.keras.initializers.GlorotUniform()([self.y_dim, 1]), name='A0')
        self.A = tf.Variable(tf.keras.initializers.GlorotUniform()([self.y_dim, self.y_dim]), name='A')

        self.trainable_variables = [self.A0, self.A] + self.nn_c.trainable_variables + self.nn_phi.trainable_variables \
                                   + self.nn_eta.trainable_variables + self.nn_xi.trainable_variables

    def nn_c_fn(self):

        inputs = tf.keras.Input(shape=(self.max_word_len, self.c_dim))

        outputs = tf.keras.layers.LSTM(units=self.h_c_dim, return_sequences=True)(inputs)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def nn_x_fn(self, **kwargs):

        inputs = tf.keras.Input(shape=(None, self.x_dim + self.h_c_dim,))

        h = inputs

        for d in range(kwargs['depth']):
            h = tf.keras.layers.Dense(units=kwargs['units'], activation=kwargs['activation'])(h)

            if kwargs['skip']:
                h = tf.keras.layers.Concatenate()([h, inputs])

        outputs = tf.keras.layers.Dense(units=self.y_dim)(h)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def log_p_y_given_x(self, y, x, x_c, mask, lens_c):

        N = x.shape[0]

        # Compute log potential

        y0 = y[:, 0]  # (N, Y)

        log_psi_0 = tf.matmul(y0, self.A0)  # (N, 1)

        y_tm1 = y[:, :-1]  # (N, T-1, Y)
        y_t = y[:, 1:]  # (N, T-1, Y)

        A_y_tm1 = tf.tensordot(y_tm1, self.A, axes=1)  # (N, T-1, Y)

        log_psi = tf.reduce_sum(A_y_tm1 * y_t, axis=2)  # (N, T-1)
        log_psi = tf.concat([log_psi_0, log_psi], axis=1)  # (N, T)

        h_c = tf.reshape(self.nn_c(tf.reshape(x_c, (N * self.max_steps, self.max_word_len, self.c_dim))),
                         (N, self.max_steps, self.max_word_len, self.h_c_dim))  # (N, T, L, HC)
        h_c = tf.squeeze(tf.gather(h_c, tf.expand_dims(lens_c, 2), batch_dims=2))  # (N, T, HC)

        h_x = tf.concat([x, h_c], axis=2)  # (N, T, X+HC)

        log_phi_all = self.nn_phi(h_x[:, :-1])  # (N, T-1, Y)
        log_eta_all = self.nn_eta(h_x)  # (N, T, Y)
        log_xi_all = self.nn_xi(h_x[:, 1:])  # (N, T-1, Y)

        log_phi = tf.reduce_sum(y[:, 1:] * log_phi_all, axis=2)  # (N, T-1)
        log_eta = tf.reduce_sum(y * log_eta_all, axis=2)  # (N, T)
        log_xi = tf.reduce_sum(y[:, :-1] * log_xi_all, axis=2)  # (N, T-1)

        log_pot = tf.reduce_sum(mask * (log_psi + log_eta), axis=1) + \
                  tf.reduce_sum(mask[:, 1:] * (log_phi + log_xi), axis=1)  # (N)

        # Compute log normaliser

        log_psi_0_all = tf.transpose(self.A0)  # (1, Y)
        log_psi_all = tf.expand_dims(self.A, axis=0)  # (1, Y, Y)

        log_alpha_init = log_psi_0_all + log_eta_all[:, 0]  # (N, Y)

        def cond(t, log_alpha_tm1):
            return tf.less(t, self.max_steps)

        def body(t, log_alpha_tm1):
            log_alpha_tm1 = tf.expand_dims(log_alpha_tm1, axis=2)  # (N, Y, 1)

            log_phi_t = log_phi_all[:, tf.cast(t - 1, tf.int32)]  # (N, Y)
            log_eta_t = log_eta_all[:, tf.cast(t, tf.int32)]  # (N, Y)
            log_xi_t = log_xi_all[:, tf.cast(t - 1, tf.int32)]  # (N, Y)

            log_alpha_t = log_phi_t + log_eta_t + tf.reduce_logsumexp(log_alpha_tm1 + log_psi_all +
                                                                      tf.expand_dims(log_xi_t, axis=2),
                                                                      axis=1)  # (N, Y)

            mask_t = tf.expand_dims(mask[:, tf.cast(t, tf.int32)], 1)  # (N, 1)

            log_alpha = tf.where(tf.equal(mask_t, 1.), log_alpha_t, tf.squeeze(log_alpha_tm1, 2))  # (N, Y)

            return t + 1., log_alpha

        loop_vars = (tf.constant(1.), log_alpha_init)

        _, log_alpha_final = tf.while_loop(cond, body, loop_vars)

        log_norm = tf.reduce_logsumexp(log_alpha_final, axis=1)  # (N)

        # Return log(p(y|x))

        return log_pot - log_norm

    def log_p_x(self, x, mask, emb_matrix, char_matrix, lens_char_matrix):

        N = x.shape[0]
        V = emb_matrix.shape[0]

        log_psi_0 = tf.transpose(self.A0)  # (1, Y)
        log_psi = tf.expand_dims(self.A, 0)  # (1, Y, Y)

        h_c_all = self.nn_c(char_matrix)  # (V, L, HC)
        h_c_all = tf.squeeze(tf.gather(h_c_all, tf.expand_dims(lens_char_matrix, 1), batch_dims=1))  # (V, HC)

        h_x_all = tf.concat([emb_matrix, h_c_all], axis=1)  # (V, X + HC)

        log_phi_all = self.nn_phi(h_x_all)  # (V, Y)
        log_eta_all = self.nn_eta(h_x_all)  # (V, Y)
        log_xi_all = self.nn_xi(h_x_all)  # (V, Y)

        log_phi = tf.gather(log_phi_all, x[:, :-1])  # (N, T-1, Y)
        log_eta = tf.gather(log_eta_all, x)  # (N, T, Y)
        log_xi = tf.gather(log_xi_all, x[:, 1:])  # (N, T-1, Y)

        log_alpha_y_init = log_psi_0 + log_eta[:, 0]  # (N, Y)
        log_alpha_x_init = log_psi_0 + log_eta_all  # (V, Y)
        log_alpha_x_out_init = tf.tile(tf.reshape(tf.reduce_logsumexp(log_alpha_x_init), (1,)), (N,))  # (N)

        def step_fn(t, log_alpha_y_tm1, log_alpha_x_tm1, log_alpha_x_out_tm1,
                    log_psi, log_phi, log_eta, log_xi, log_phi_all, log_eta_all, log_xi_all):

            log_alpha_y_tm1 = tf.expand_dims(log_alpha_y_tm1, axis=2)  # (N, Y, 1)
            log_alpha_x_tm1 = tf.expand_dims(log_alpha_x_tm1, axis=2)  # (V, Y, 1)

            log_phi_t = log_phi[:, tf.cast(t - 1, tf.int32)]  # (N, Y)
            log_eta_t = log_eta[:, tf.cast(t, tf.int32)]  # (N, Y)
            log_xi_t = log_xi[:, tf.cast(t - 1, tf.int32)]  # (N, Y)

            log_alpha_y_t = log_phi_t + log_eta_t + tf.reduce_logsumexp(log_alpha_y_tm1 + log_psi +
                                                                        tf.expand_dims(log_xi_t, axis=2),
                                                                        axis=1)  # (N, Y)

            mask_t = tf.expand_dims(mask[:, tf.cast(t, tf.int32)], 1)  # (N, 1)

            log_alpha_y = tf.where(tf.equal(mask_t, 1.), log_alpha_y_t, tf.squeeze(log_alpha_y_tm1, 2))  # (N, Y)

            log_sum_x = tf.reduce_logsumexp(log_alpha_x_tm1 + tf.reshape(log_phi_all, (V, 1, self.y_dim)), axis=0)
            # (Y, Y)

            log_alpha_x = log_eta_all + tf.reduce_logsumexp(log_psi + tf.expand_dims(log_xi_all, 2) +
                                                            tf.expand_dims(log_sum_x, 0), axis=1)  # (V, Y)

            log_alpha_x_out = tf.where(tf.equal(tf.squeeze(mask_t, 1), 1.), tf.reduce_logsumexp(log_alpha_x),
                                       log_alpha_x_out_tm1)  # (N)

            return log_alpha_y, log_alpha_x, log_alpha_x_out

        def cond(t, log_alpha_y_tm1, log_alpha_x_tm1, log_alpha_x_out_tm1):
            return tf.less(t, self.max_steps)

        def body(t, log_alpha_y_tm1, log_alpha_x_tm1, log_alpha_x_out_tm1):

            log_alpha_y, log_alpha_x, log_alpha_x_out = step_fn(t, log_alpha_y_tm1, log_alpha_x_tm1,
                                                                log_alpha_x_out_tm1,
                                                                log_psi, log_phi, log_eta, log_xi,
                                                                log_phi_all, log_eta_all, log_xi_all)

            return t + 1., log_alpha_y, log_alpha_x, log_alpha_x_out

        loop_vars = (tf.constant(1.), log_alpha_y_init, log_alpha_x_init, log_alpha_x_out_init)

        _, log_alpha_y_final, _, log_alpha_x_final = tf.while_loop(cond, body, loop_vars)

        log_alpha_y_final = tf.reduce_logsumexp(log_alpha_y_final, axis=1)  # (N)

        log_p_x = log_alpha_y_final - log_alpha_x_final  # (N)

        return log_p_x

    def viterbi(self, x, x_c, mask, lens_c):

        N = x.shape[0]

        log_psi_0_all = tf.transpose(self.A0)  # (1, Y)
        log_psi_all = tf.expand_dims(self.A, 0)  # (1, Y, Y)

        h_c = tf.reshape(self.nn_c(tf.reshape(x_c, (N * self.max_steps, self.max_word_len, self.c_dim))),
                         (N, self.max_steps, self.max_word_len, self.h_c_dim))  # (N, T, L, HC)
        h_c = tf.squeeze(tf.gather(h_c, tf.expand_dims(lens_c, 2), batch_dims=2))  # (N, T, HC)

        h_c = tf.cond(pred=tf.equal(tf.rank(h_c), 2), true_fn=lambda: tf.expand_dims(h_c, 0), false_fn=lambda: h_c)

        h_x = tf.concat([x, h_c], axis=2)  # (N, T, X + HC)

        log_phi_all = self.nn_phi(h_x[:, :-1])  # (N, T-1, Y)
        log_eta_all = self.nn_eta(h_x)  # (N, T, Y)
        log_xi_all = self.nn_xi(h_x[:, 1:])  # (N, T-1, Y)

        log_alpha_init = log_psi_0_all + log_eta_all[:, 0]  # (N, Y)
        log_alpha_max_init = log_alpha_init  # (N, Y)
        argmax_init = tf.zeros((N, self.y_dim), dtype=tf.int64)  # (N, Y)

        def step_fwd(a, t):
            log_alpha_tm1, log_alpha_max_tm1, _ = a

            log_alpha_tm1 = tf.expand_dims(log_alpha_tm1, axis=2)  # (N, Y, 1)
            log_alpha_max_tm1 = tf.expand_dims(log_alpha_max_tm1, axis=2)  # (N, Y, 1)

            log_phi_t = log_phi_all[:, t-1]  # (N, Y)
            log_eta_t = log_eta_all[:, t]  # (N, Y)
            log_xi_t = log_xi_all[:, t-1]  # (N, Y)

            log_alpha_t = log_phi_t + log_eta_t + tf.reduce_logsumexp(log_alpha_tm1 + log_psi_all +
                                                                      tf.expand_dims(log_xi_t, axis=2),
                                                                      axis=1)  # (N, Y)

            argmax_t = tf.argmax(log_alpha_max_tm1 + log_psi_all + tf.expand_dims(log_xi_t, axis=2), axis=1)  # (N, Y)

            log_alpha_max_t = log_phi_t + log_eta_t + tf.reduce_max(log_alpha_max_tm1 + log_psi_all +
                                                                    tf.expand_dims(log_xi_t, axis=2),
                                                                    axis=1)  # (N, Y)

            mask_t = tf.expand_dims(mask[:, t], 1)  # (N, 1)

            log_alpha = tf.where(tf.equal(mask_t, 1.), log_alpha_t, tf.squeeze(log_alpha_tm1, 2))  # (N, Y)

            log_alpha_max = tf.where(tf.equal(mask_t, 1.), log_alpha_max_t, tf.squeeze(log_alpha_max_tm1, 2))  # (N, Y)

            argmax = tf.where(tf.equal(mask_t, 1.), argmax_t, tf.argmax(log_alpha_max_tm1, axis=1))  # (N, Y)

            return log_alpha, log_alpha_max, argmax

        log_alpha_all, log_alpha_max_all, argmax_all = tf.scan(step_fwd,
                                                               elems=tf.range(1, self.max_steps, 1),
                                                               initializer=(log_alpha_init, log_alpha_max_init,
                                                                            argmax_init),
                                                               )

        log_norm = tf.reduce_logsumexp(log_alpha_all[-1], axis=1)  # (N)

        y_init = tf.argmax(log_alpha_max_all[-1], axis=1)  # (N)

        def step_bwd(a, t):
            y_tm1 = a

            y_t = tf.reshape(tf.gather_nd(argmax_all[t], tf.expand_dims(y_tm1, 1), batch_dims=1), [-1])  # (N)

            return y_t

        y = tf.scan(step_bwd,
                    elems=tf.range(self.max_steps-2, -1, -1),
                    initializer=y_init,
                    )

        y = tf.concat([tf.expand_dims(y_init, 0), y], axis=0)  # (T, N)

        y = tf.transpose(y[::-1])  # (N, T)

        # Compute log potential

        y_one_hot = tf.one_hot(y, depth=self.y_dim)  # (N, T, Y)

        y0 = y_one_hot[:, 0]  # (N, Y)

        log_psi_0 = tf.matmul(y0, self.A0)  # (N, 1)

        y_tm1 = y_one_hot[:, :-1]  # (N, T-1, Y)
        y_t = y_one_hot[:, 1:]  # (N, T-1, Y)

        A_y_tm1 = tf.tensordot(y_tm1, self.A, axes=1)  # (N, T-1, Y)

        log_psi = tf.reduce_sum(A_y_tm1 * y_t, axis=2)  # (N, T-1)
        log_psi = tf.concat([log_psi_0, log_psi], axis=1)  # (N, T)

        log_phi = tf.reduce_sum(y_t * log_phi_all, axis=2)  # (N, T-1)
        log_eta = tf.reduce_sum(y_one_hot * log_eta_all, axis=2)  # (N, T)
        log_xi = tf.reduce_sum(y_tm1 * log_xi_all, axis=2)  # (N, T)

        log_pot = tf.reduce_sum(mask * (log_psi + log_eta), axis=1) + \
                  tf.reduce_sum(mask[:, 1:] * (log_phi + log_xi), axis=1)  # (N)

        return y, log_pot - log_norm
