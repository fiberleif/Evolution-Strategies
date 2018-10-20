'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''
import os
import ray
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from filter import get_filter

class Policy(object):
    # base class for policy
    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.state_normalize = {"mean", None, "std", None}
        self.update_filter = True

    def get_observation_filter(self):
        return self.observation_filter

    def set_state_normalize(self, mean, std):
        self.state_normalize["mean"] = mean
        self.state_normalize["std"] = std

    def update_weights(self, new_weights):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def get_weights_plus_stats(self):
        raise  NotImplementedError

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

class MLPPolicy(object):
    """
    MLP policy class using tensorflow module
    """
    def __init__(self, name, obs_dim, nb_actions, layer_norm, activation, layer_num, layer_width, save_path):
        self.name = name
        self.state_normalize = {"mean": None, "std": None}
        self.obs_dim = obs_dim
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.activation = activation
        self.layer_num = layer_num
        self.layer_width = layer_width
        self.save_path = save_path
        self._build_graph()
        self.init = tf.global_variables_initializer()
        self._make_session()
        self.variables = ray.experimental.TensorFlowVariables(self.action, self.sess)

    def set_state_normalize(self, mean, std):
        self.state_normalize["mean"] = mean
        self.state_normalize["std"] = std

    def _build_graph(self):
        with tf.variable_scope(self.name) as scope:
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
            x = self.obs_ph
            for _ in range(self.layer_num):
                x = tf.layers.dense(x, self.layer_width)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = self.activation(x)

            x = tf.layers.dense(x, self.nb_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)

            self.action = x = (x + 1.0) / 2.0

    def _make_session(self):
        self.sess = tf.Session()
        self.sess.run(self.init)

    def update_weights(self, new_weights):
        self.variables.set_weights(new_weights)

    def get_weights(self):
        return self.variables.get_weights()

    def save_weights_plus_stats(self):
        if not (os.path.exists("es_model")):
            os.makedirs("es_model")
        self.saver = tf.train.Saver(self.trainable_vars)
        self.saver.save(self.sess, "./es_model/model.ckpt")

        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([mu, std])
        np.savez(self.save_path + "./obs_rms", aux)

    # def load_weights_plus_stats(self):
    #     stats = np.load('./obs_rms.npz')
    #     stats = stats.items()[0][1]

    def act(self, ob):
        ob = np.array(ob).reshape(1, len(ob))
        normalized_ob = (ob - self.state_normalize["mean"]) / self.state_normalize["std"]
        action = self.sess.run(self.action, feed_dict={self.obs_ph: normalized_ob})
        return action

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if "LayerNorm" not in var.name]
