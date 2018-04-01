""" Classes wrapping the OpenAI Gym.
"""
from collections import deque

import torch
import numpy as np
from termcolor import colored as clr
import gym

from . import transformations as T


__all__ = ["TorchWrapper", "SqueezeRewards", "FrameStack", "DoneAfterLostLife",
           "TransformObservations", "get_wrapped_atari"]


class TorchWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, o):
        """ Convert from numpy to torch.
            Also change from (h, w, c*hist) to (batch, hist*c, h, w)
        """
        return torch.from_numpy(o).permute(2, 0, 1).unsqueeze(0).contiguous()


class SqueezeRewards(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        print("[Reward Wrapper] for clamping rewards to -+1")

    def reward(self, reward):
        return float(np.sign(reward))


class TransformObservations(gym.ObservationWrapper):
    def __init__(self, env, transformations):
        super().__init__(env)
        self.transformations = transformations

        for t in self.transformations:
            try:
                t.update_env_specs(self)
            except AttributeError:
                pass

    def observation(self, o):
        for t in self.transformations:
            o = t.transform(o)
        return o

    def _reset(self):
        o = self.env.reset()
        return self.observation(o)

    def __str__(self):
        r = ""
        for t in self.transformations:
            r += str(t)
        return '\n<{}({})\n{}>'.format(type(self).__name__, r, self.env)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        """
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shape = (shp[0], shp[1], shp[2] * k)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape,
                                                dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=2)


class DoneAfterLostLife(gym.Wrapper):
    def __init__(self, env):
        super(DoneAfterLostLife, self).__init__(env)

        self.no_more_lives = True
        self.crt_live = env.unwrapped.ale.lives()
        self.has_many_lives = self.crt_live != 0

        if self.has_many_lives:
            self.step = self._many_lives_step
        else:
            self.step = self._one_live_step

        not_a = clr("not a", attrs=['bold'])
        print("[DoneAfterLostLife Wrapper]  %s is %s many lives game."
              % (env.env.spec.id, "a" if self.has_many_lives else not_a))

    def reset(self):
        if self.no_more_lives:
            obs = self.env.reset()
            self.crt_live = self.env.unwrapped.ale.lives()
            return obs
        return self.__obs

    def _many_lives_step(self, action):
        obs, reward, done, info = self.env.step(action)
        crt_live = self.env.unwrapped.ale.lives()
        if crt_live < self.crt_live:
            # just lost a live
            done = True
            self.crt_live = crt_live

        if crt_live == 0:
            self.no_more_lives = True
        else:
            self.no_more_lives = False
            self.__obs = obs
        return obs, reward, done, info

    def _one_live_step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


def get_wrapped_atari(env_name, mode, **kwargs):
    """ The preprocessing traditionally used by DeepMind on Atari.
    """

    env = gym.make(env_name)
    env = FireResetEnv(env)

    if mode == 'training':
        env = DoneAfterLostLife(env)
        env = SqueezeRewards(env)

    env = TransformObservations(env, [
        T.Downsample(84, 84),
        T.RGB2Y(),
        T.Normalize()
    ])

    hist_len = kwargs['hist_len'] if 'hist_len' in kwargs else 4
    env = FrameStack(env, hist_len)
    env = TorchWrapper(env)
    return env


if __name__ == "__main__":
    pass
