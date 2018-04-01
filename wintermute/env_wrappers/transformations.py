from abc import ABC, abstractmethod

import gym
import lycon
import numpy as np


__all__ = ["Downsample", "Normalize", "RGB2Y"]


class AbstractTransformation(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, observation):
        pass

    def __str__(self):
        return '[{}]'.format(type(self).__name__)


class Downsample(AbstractTransformation):
    nearest = lycon.Interpolation.NEAREST
    linear = lycon.Interpolation.LINEAR
    cubic = lycon.Interpolation.CUBIC
    area = lycon.Interpolation.AREA
    lanczos = lycon.Interpolation.LANCZOS

    def __init__(self, height, width, interpolation=linear):
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation = interpolation
        if interpolation is lycon.Interpolation.LANCZOS:
            print("Warning, Lanczos interpolation can be slow.")

    def update_env_specs(self, env):
        obs_space = env.observation_space
        low, high, dtype = obs_space.low, obs_space.high, obs_space.dtype
        low = np.resize(low, (self.height, self.width, 3))
        high = np.resize(high, (self.height, self.width, 3))
        env.observation_space = gym.spaces.Box(low, high, dtype=dtype)

    def transform(self, o):
        return lycon.resize(o, width=self.width, height=self.height,
                            interpolation=self.interpolation)


class RGB2Y(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.rgb = np.array([.2126, .7152, .0722], dtype=np.float32)

    def update_env_specs(self, env):
        obs_space = env.observation_space
        low, high, dtype = obs_space.low, obs_space.high, obs_space.dtype
        shape = (low.shape[0], low.shape[1], 1)
        low, high = np.resize(low, shape), np.resize(high, shape)
        env.observation_space = gym.spaces.Box(low, high, dtype=dtype)

    def transform(self, o):
        dtype = o.dtype
        shape = (o.shape[0], o.shape[1], 1)
        return np.dot(o, self.rgb).reshape(shape).astype(dtype)


class Normalize(AbstractTransformation):
    def __init__(self):
        super().__init__()

    def update_env_specs(self, env):
        shape = env.observation_space.low.shape
        env.observation_space = gym.spaces.Box(low=0, high=1, shape=shape,
                                               dtype=np.float32)

    def transform(self, o):
        return np.array(o).astype(np.float32) / 255.0


if __name__ == '__main__':
    from wrappers import SqueezeRewards, TransformObservations

    env = gym.make('SpaceInvaders-v0')
    env = TransformObservations(env, [
            Downsample(84, 84),
            RGB2Y(),
            Normalize()
        ])
    env = SqueezeRewards(env)

    o, d = env.reset(), False
    print(o.shape, o.dtype, "max=%3.2f" % o.max(), "min=%3.2f" % o.min(),
          "mean=%3.2f" % o.mean())
    print(env.observation_space, env.unwrapped.observation_space)
    print(env)

    """
    i = 0
    while not d:
        o, r, d, _ = env.step(env.action_space.sample())
        print(i, o.shape, o.dtype, o.max(), o.min(), "%3.2f" % o.mean())
        print(env.observation_space, env.unwrapped.observation_space)
        if d:
            o, d = env.reset(), False
            i += 1
        if i == 2:
            break
    """
