import tensorflow as tf
import tensorflow.contrib as tc
import ray

class Actor(object):
    def __init__(self, name, nb_actions, layer_norm, activation, layer_num, layer_width):
        self.name = name
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.activation = activation
        self.layer_num = layer_num
        self.layer_width = layer_width
        self._build_graph()
        self.init = tf.global_variables_initializer()
        self._make_session()

    def _build_graph(self):
        with tf.variable_scope(self.name) as scope:
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, 200))
            for _ in range(self.layer_num):
                x = tf.layers.dense(self.obs_ph, self.layer_width)
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

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if "LayerNorm" not in var.name]


if __name__ == "__main__":
    actor = Actor("policy", 19, True, tf.nn.relu, 2, 64)
    # print(actor.vars)
    # print(actor.trainable_vars)
    # print(actor.perturbable_vars)
    variables = ray.experimental.TensorFlowVariables(actor.action, actor.sess)
    weights = variables.get_weights()
    print(weights)