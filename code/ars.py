'''
Distributional implementation of Evolution Strategies.
'''

# utility module
import time
import os
import utils
import numpy as np
import logz
import tensorflow as tf
import copy

# module for argument parsing
import argparse

# module for env making
import gym
from osim.env import ProstheticsEnv

# module for distributed computing
import ray
import socket

# module for algorithm logic
import optimizers
from policies import *
from shared_noise import *
from observation import ObsProcessWrapper, RewardReshapeWrapper, DictToListFull

env_type_choices = ["MuJoCo", "Atari", "Prosthetics"]
env_difficulty_choices = [1, 2]
policy_type_choices = ["Linear", "MLP"]


@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_params=None,
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        if (env_params["type"] == "MuJoCo"):
            self.env = gym.make(env_params["name"])
            self.env.seed(env_seed)
        elif (env_params["type"] == "Prosthetics"):
            env = ProstheticsEnv(visualize=False, difficulty=env_params['difficulty'])
            env = ObsProcessWrapper(env)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params

        if policy_params['type'] == 'Linear':
            self.policy = LinearPolicy(policy_params)
        else:
            self.policy = MLPPolicy("policy", policy_params["obs_dim"], policy_params["ac_dim"], policy_params["layer_norm"], tf.nn.selu, policy_params["layer_depth"],\
                                    policy_params["layer_width"], None)
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length

        
    # def get_weights_plus_stats(self):
    #     """
    #     Get current policy weights and current statistics of past states.
    #     """
    #     assert self.policy_params['type'] == 'linear'
    #     return self.policy.get_weights_plus_stats()
    

    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
            
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx = [], {}
        for name in w_policy.keys():
            if "LayerNorm" not in name:
                deltas_idx[name] = []
        steps = 0

        for i in range(num_rollouts):
            if evaluate:
                self.policy.update_weights(w_policy)
                # deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps = self.rollout(shift = 0.)
                rollout_rewards.append(reward)
                
            else:
                pos_w_policy = copy.copy(w_policy)
                neg_w_policy = copy.copy(w_policy)
                for name, weight in w_policy.items():
                    if "LayerNorm" not in name:
                        idx, delta = self.deltas.get_delta(weight.size)

                        delta = (self.delta_std * delta).reshape(weight.shape)
                        # deltas_idx.append(idx)
                        deltas_idx[name].append(idx)
                        pos_w_policy[name] = weight + delta
                        neg_w_policy[name] = weight - delta
                        # set to true so that state statistics are updated
                        # self.policy.update_filter = True

                        # compute reward and number of timesteps used for positive perturbation rollout
                        # self.policy.update_weights(weight + delta)

                self.policy.update_weights(pos_w_policy)
                pos_reward, pos_steps  = self.rollout(shift = shift)
                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(neg_w_policy)
                neg_reward, neg_steps = self.rollout(shift = shift)
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    
class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_params=None,
                 policy_params=None,
                 num_workers=16,
                 num_deltas=60,
                 deltas_used=60,
                 delta_std=0.003,
                 logdir=None,
                 model_path=None,
                 save_path=None,
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123):

        logz.configure_output_dir(logdir)
        logz.save_params(params)

        self.timesteps = 0
        self.ob_size = policy_params["ob_dim"]
        self.action_size = policy_params["ac_dim"]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.model_path = model_path
        self.save_path = save_path
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_params=env_params,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]


        # initialize policy 
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)

        else:
            self.policy = MLPPolicy("policy", policy_params["obs_dim"], policy_params["ac_dim"], policy_params["layer_norm"], tf.nn.selu, policy_params["layer_depth"],\
                                    policy_params["layer_width"], self.save_path)
            # load model
            self.load_model()
        self.w_policy = self.policy.get_weights()
        # initialize optimization algorithm
        # self.optimizer = optimizers.SGD(self.w_policy, self.step_size)
        print("Initialization of ARS complete.")

    def load_model(self):
        """
        load policy model
        """
        print("Start loading models...")
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph(self.model_path + '.meta')
        new_saver.restore(sess, self.model_path)
        ops = tf.get_collection('eval')
        variables = ray.experimental.TensorFlowVariables(ops[2], sess)
        weights = variables.get_weights()
        actor_weights = {}
        obs_weights = {}
        for name, weight in weights.items():
            if "actor" in name:
                actor_weights[name.replace("actor", "policy")] = weight
            else:
                obs_weights[name] = weight
        self.policy.update_weights(actor_weights)

        runnningsum = obs_weights['obs_rms/runningsum']
        runningsumsq = obs_weights['obs_rms/runningsumsq']
        runningcount = obs_weights['obs_rms/count']
        mean = runnningsum / runningcount
        std = np.sqrt(np.maximum((runningsumsq / runningcount) - np.square(mean), 1e-2))
        self.policy.set_state_normalize(mean, std)

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        # put policy weights in the object store
        self.w_policy = self.policy.get_weights()
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
            
        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], {}
        for name in self.w_policy.keys():
            if "LayerNorm" not in name:
                deltas_idx[name] = []

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            for name in deltas_idx.keys():
                deltas_idx[name] += result['deltas_idx'][name]
            rollout_rewards += result['rollout_rewards']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            for name in deltas_idx.keys():
                deltas_idx[name] += result['deltas_idx'][name]
            rollout_rewards += result['rollout_rewards']

        # deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # # select top performing directions if deltas_used < num_deltas
        # max_rewards = np.max(rollout_rewards, axis = 1)
        # if self.deltas_used > self.num_deltas:
        #     self.deltas_used = self.num_deltas
        #
        # idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        # deltas_idx = deltas_idx[idx]
        # rollout_rewards = rollout_rewards[idx,:]
        
        # normalize rewards by their standard deviation
        # rollout_rewards /= np.std(rollout_rewards)
        rollout_rewards = self.compute_centered_ranks(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        for name in deltas_idx.keys():
            g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                      (self.deltas.get(idx, self.w_policy[name].size)
                                                       for idx in deltas_idx[name]),
                                                      batch_size = 500)
            g_hat /= len(deltas_idx[name])
            self.w_policy[name] += g_hat * self.step_size
        self.policy.update_weights(self.w_policy)
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)

        # print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        # self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)

    def compute_ranks(self, x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def compute_centered_ranks(self, x):
        y = self.compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        # g_hat = self.aggregate_rollouts()
        # print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        # self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        self.aggregate_rollouts()
        return

    def train(self, num_iter):
        start = time.time()
        for i in range(num_iter):
            self.evaluate(start, i)
            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)           
            print('iter ', i,' done')

            # record statistics every 10 iterations
            # if ((i + 1) % 1 == 0):

            # t1 = time.time()
            # # get statistics from all workers
            # for j in range(self.num_workers):
            #     self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            # self.policy.observation_filter.stats_increment()
            #
            # # make sure master filter buffer is clear
            # self.policy.observation_filter.clear_buffer()
            # # sync all workers
            # filter_id = ray.put(self.policy.observation_filter)
            # setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # # waiting for sync of all workers
            # ray.get(setting_filters_ids)
            #
            # increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # # waiting for increment of all workers
            # ray.get(increment_filters_ids)
            # t2 = time.time()
            # print('Time to sync statistics:', t2 - t1)
                        
        return

    def evaluate(self, start, i):

        rewards = self.aggregate_rollouts(num_rollouts=2, evaluate=True)
        # w = ray.get(self.workers[0].get_weights_plus_stats.remote())
        # np.savez(self.logdir + "/lin_policy_plus", w)

        print(sorted(self.params.items()))
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", i + 1)
        logz.log_tabular("AverageReward", np.mean(rewards))
        logz.log_tabular("StdRewards", np.std(rewards))
        logz.log_tabular("MaxRewardRollout", np.max(rewards))
        logz.log_tabular("MinRewardRollout", np.min(rewards))
        logz.log_tabular("timesteps", self.timesteps)
        logz.dump_tabular()


def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    if params['env_type'] == "MuJoCo":
        env = gym.make(params['env_name'])
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        env_params = {'type': params['env_type'],
                      'name': params['env_name']}
    elif params['env_type'] == "Prosthetics":
        env = ProstheticsEnv(visualize=False, difficulty=params['env_difficulty'])
        env = ObsProcessWrapper(env)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        env_params = {'type': params['env_type'],
                      'difficulty': params['env_difficulty']}
    else:
        raise NotImplementedError

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    if params['policy_type'] == "Linear":
        policy_params = {'type': 'linear',
                         'ob_filter': params['filter'],
                         'ob_dim': ob_dim,
                         'ac_dim': ac_dim}
    else:
        policy_params = {'type': 'MLP',
                         'ob_filter': params['filter'],
                         'ob_dim': ob_dim,
                         'ac_dim': ac_dim,
                         'layer_norm': params['layer_norm'],
                         'layer_depth': params['layer_depth'],
                         'layer_width': params['layer_width']}


    ARS = ARSLearner(env_params=env_params,
                     policy_params=policy_params,
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     model_path = params['model_path'],
                     save_path = params['save_path'],
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'])
        
    ARS.train(params['n_iter'])
       
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    bool_mapper = lambda str: True if 'True' in str or 'true' in str else False
    parser.add_argument('--env_type', choices=env_type_choices, default="Prosthetics")

    # ------ arguments used when env_type = "MuJoCo" or env_type = "Atari"
    parser.add_argument('--env_name', type=str, default="HalfCheetah-v1")
    # ------

    # ------ arguments used when env_type = "Prosthetics"
    parser.add_argument('--env_difficulty', choices=env_difficulty_choices, default=1)
    # ------

    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.0002)
    parser.add_argument('--n_workers', '-e', type=int, default=4)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_type', choices=policy_type_choices, default="MLP")

    # ------ arguments used when policy_type = "MLP"
    parser.add_argument('--layer_norm', type=bool_mapper, default=True)
    parser.add_argument('--layer_depth', type=int, default=2)
    parser.add_argument('--layer_width', type=int, default=128)
    # ------

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    parser.add_argument('--dir_path', type=str, default='log')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    # local_ip = socket.gethostbyname(socket.gethostname())
    # ray.init(redis_address= local_ip + ':6379')
    ray.init()
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)

