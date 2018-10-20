import tensorflow as tf
import ray
from policies import MLPPolicy

def load_model(model_path):
    print("Start loading models...")
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(model_path + '.meta')
    new_saver.restore(sess, model_path)
    ops = tf.get_collection('eval')
    variables = ray.experimental.TensorFlowVariables(ops[2], sess)
    weights = variables.get_weights()
    # print(weights)
    actor_weights = {}
    for name, weight in weights.items():
        if "obs" in name:
            # print(name, weight.shape)
            print(name)
            actor_weights[name.replace("actor", "policy")] = weight

    actor = MLPPolicy("policy", 206, 19, True, tf.nn.selu, 2, 128, "./log/")
    # print([ i.name for i in actor.vars])
    actor.update_weights(actor_weights)
    actor_weights_after_set = actor.get_weights()
    for weight, weight_after_set in zip(actor_weights.values(), actor_weights_after_set.values()):
        print((weight==weight_after_set).all())

model_path = "../model/DDPG_layer-width-128_activation-selu_seed-1-2018-10-09-05-12-44-1/best"
load_model(model_path)