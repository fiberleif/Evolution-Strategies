import gym
import numpy as np

balance_pos = {'femur_r': np.array([-0.55308777, -0.51710186,  0.65322247]),
 'pros_tibia_r': np.array([-0.15818479, -0.97166179,  0.17564435]),
 'pros_foot_r': np.array([-0.08365099, -0.99215679,  0.09288392]),
 'femur_l': np.array([-0.55308777, -0.51710186, -0.65322247]),
 'tibia_l': np.array([-0.15818479, -0.97166179, -0.17564435]),
 'talus_l': np.array([-0.08365099, -0.99215679, -0.09288392]),
 'calcn_l': np.array([-0.13097871, -0.98666875, -0.09658859]),
 'toes_l': np.array([ 0.05820437, -0.99346389, -0.09819242]),
 'torso': np.array([-0.77731677,  0.6291094 ,  0.        ]),
 'head': np.array([-0.08353876,  0.99650453,  0.        ])}

# constant speed
speed = [1.25, 0, 0]

def cross_product(a, b):
    return a[0] * b[1] - a[1] * b[0]

class ConstantRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ConstantRewardWrapper, self).__init__(env)
        self.osim_model = self.env.osim_model
        self.time_limit = self.env.time_limit

    def _reward(self):
        state_desc = self.env.get_state_desc()
        prev_state_desc = self.env.get_prev_state_desc()
        penalty = 0

        # Small penalty for too much activation (cost of transport)
        penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001

        # Big penalty for not matching the vector on the X,Z projection.
        # No penalty for the vertical axis
        # penalty += (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) ** 2
        # penalty += (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) ** 2
        penalty += (state_desc["body_vel"]["pelvis"][0] - speed[0]) ** 2
        penalty += (state_desc["body_vel"]["pelvis"][2] - speed[2]) ** 2

        # Reward for not falling
        reward = 10.0
        return reward - penalty

    def step(self, action, project=False):
        obs, reward, done, info = self.env.step(action, project=project)
        reward = self._reward()
        return obs, reward, done, info

    def reset(self, project=False):
        obs = self.env.reset(project=project)
        return obs

class ObsProcessWrapper(gym.Wrapper):
    def __init__(self, env, add_feature, round, **kwargs):
        self.add_feature = add_feature
        self.round = round
        super(ObsProcessWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=-float('Inf'),
                                                high=float('Inf'),
                                                shape=(206,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-float('0.0'),
                                                high=float('1.0'),
                                                shape=(19,),
                                                dtype=np.float32)
    def obs_process(self, state_desc):
        res = []
        pelvis = None

        for body_part in ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l',
                          'calcn_l', 'toes_l', 'torso', 'head']:
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
            else:
                cur_upd = cur
                relative_pos = [cur[i] - pelvis[i] for i in range(2)]
                relative_vel = [cur[i] - pelvis[i] for i in range(2, 4)]
                relative_acc = [cur[i] - pelvis[i] for i in range(4, 6)]
                cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6, 7)]
                # cur_upd[:] = [cur[i] - pelvis[i] for i in range(7)]
                # compute extra observation
                ta = cross_product(balance_pos[body_part], relative_pos) / np.linalg.norm(relative_pos)
                tb = cross_product(relative_acc, relative_pos) / np.linalg.norm(relative_pos)
                res += cur
                if self.add_feature:
                    res += [ta * tb]  # negative ta * tb tends to fail.

        for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]
        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        if self.round == 2:
            res += state_desc["target_vel"]
            res += state_desc['body_pos']['pelvis'][0:2]
            res += [float(self.env.osim_model.istep) / self.env.time_limit]
        return res

    def step(self, action):
        obs, r, done, info = self.env.step(action, project=False)
        if obs is not None:
            obs = self.obs_process(obs)
        return obs, r, done, info

    def reset(self):
        obs = self.env.reset(project=False)
        obs = self.obs_process(obs)
        return obs


class RewardReshapeWrapper(gym.Wrapper):
    def __init__(self, env, bonus):
        super(RewardReshapeWrapper, self).__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        r += self.bonus
        return obs, r, done, info

    def reset(self):
        return self.env.reset()


class DictToListFull(gym.Wrapper):
    def __init__(self, env):
        """
        A wrapper that formats dict-type observation to list-type observation.
        Appends all meaningful unique numbers in the dict-type observation to a
        list. The resulting list has length 347.
        """
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-float('Inf'),
                                                high=float('Inf'),
                                                shape=(350,),
                                                dtype=np.float32)

    def reset(self):
        state_desc = self.env.reset(project=False)
        return self._dict_to_list(state_desc)

    def step(self, action):
        state_desc, reward, done, info = self.env.step(action, project=False)
        return [self._dict_to_list(state_desc), reward, done, info]

    def _dict_to_list(self, state_desc):
        """
        Return observation list of length 347 given a dict-type observation.
        For more details about the observation, visit this page:
        http://osim-rl.stanford.edu/docs/nips2018/observation/
        """
        res = []

        # Body Observations
        for info_type in ['body_pos', 'body_pos_rot',
                          'body_vel', 'body_vel_rot',
                          'body_acc', 'body_acc_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += state_desc[info_type][body_part]

        # Joint Observations
        # Neglecting `back_0`, `mtp_l`, `subtalar_l` since they do not move
        for info_type in ['joint_pos', 'joint_vel', 'joint_acc']:
            for joint in ['ankle_l', 'ankle_r', 'back', 'ground_pelvis',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += state_desc[info_type][joint]

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r',
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(state_desc['muscles'][muscle]['activation'])
            res.append(state_desc['muscles'][muscle]['fiber_force'])
            res.append(state_desc['muscles'][muscle]['fiber_length'])
            res.append(state_desc['muscles'][muscle]['fiber_velocity'])

        # Force Observations
        # Neglecting forces corresponding to muscles as they are redundant with
        # `fiber_forces` in muscles dictionaries
        for force in ['AnkleLimit_l', 'AnkleLimit_r',
                      'HipAddLimit_l', 'HipAddLimit_r',
                      'HipLimit_l', 'HipLimit_r', 'KneeLimit_l', 'KneeLimit_r']:
            res += state_desc['forces'][force]

        # Center of Mass Observations
        res += state_desc['misc']['mass_center_pos']
        res += state_desc['misc']['mass_center_vel']
        res += state_desc['misc']['mass_center_acc']

        return res
