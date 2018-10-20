import argparse
import os
import tensorflow as tf
import numpy as np
# import baselines.common.tf_util as U
import gym
from osim.env import ProstheticsEnv
from osim.http.client import Client
from policies import *
# from baselines.common.misc_util import (
#     set_global_seeds,
#     boolean_flag,
# )
from utils import get_difficulty
from observation import ObsProcessWrapper, RewardReshapeWrapper

remote_base = ["http://grader.crowdai.org:1729", "http://grader.crowdai.org:1730"]
crowdai_token = "ac4a9c1e83a4b2e6d6d7df408b4e7519"

class OSmodel:
    ## simulate local env
    def __init__(self):
        self.istep = 0

class RemoteProstheticsEnv(gym.Env):
    def __init__(self, base, token, round):
        self.base = base
        self.token = token
        self.client = None
        ## simulate local env
        self.osim_model = OSmodel()
        self.time_limit = 300 if round == 1 else 1000

    def reset(self, project=True):
        if self.client == None:
            self.client = Client(self.base)
            obs = self.client.env_create(self.token, env_id='ProstheticsEnv')
            self.osim_model.istep = 0
            return obs
        else:
            ### It is not allowed to call reset() twice in submitting.
            raise NotImplementedError

    def step(self, action, project=True):
        self.osim_model.istep += 1
        [obs, reward, done, info] = self.client.env_step(action.tolist(), render=True)
        if done:
            self.osim_model.istep = 0
            obs = self.client.env_reset()
            if not obs:
                done = True
            else:
                done = False
        return obs, reward, done, info

def run(online, model, obs_rms, round, visualize):
    # Load Models
    print("Start loading model...")
    policy = MLPPolicy("policy", 206, 19, True, tf.nn.selu, 2, 128, None)
    saver = tf.train.Saver()
    saver.restore(policy.sess, model)
    obs = np.load(obs_rms)
    obs = list(obs.values())
    mean = obs[0][0]
    std = obs[0][1]
    policy.set_state_normalize(mean, std)
    print("Load model success.")

    # Init env (TODO: online)
    difficulty = get_difficulty(round)
    if not online:
        env = ProstheticsEnv(visualize=visualize, difficulty=difficulty)
    else:
        env = RemoteProstheticsEnv(remote_base[difficulty], crowdai_token, round)

    env = ObsProcessWrapper(env, add_feature=True, round=round)
    total_r = 0.
    print("Prepare to eval.")
    for s in range(1 if online else 2):
        obs = env.reset()
        done = False
        step = 0
        episode_r = 0.0

        # Start eval
        while not done:
            step += 1
            a = policy.act(obs)
            obs, r, done, info = env.step(a)
            if r:
                total_r += r
                episode_r += r
            else:
                break
        print('Episode Done: total reward()'.format(episode_r))
    print('Eval End: total reward {}'.format(total_r / 2))
    if online:
        env.unwrapped.client.submit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # boolean_flag(parser, 'online', default=False)
    # boolean_flag(parser, 'visualize', default=False)
    bool_mapper = lambda str: True if 'True' in str or 'true' in str else False
    parser.add_argument('--online', type=bool_mapper, default=False)
    parser.add_argument('--visualize', type=bool_mapper, default=False)
    parser.add_argument('--model', type=str, default="./es_model/model.ckpt")
    parser.add_argument('--obs_rms', type="./obs_rms")
    parser.add_argument('--round', type=int, default=1, choices=[1, 2])
    # parse
    args = parser.parse_args()
    run(args.online, args.model, args.obs_rms, args.round, args.visualize)