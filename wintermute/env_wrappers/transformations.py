from abc import ABC, abstractmethod

import gym
import lycon
import numpy as np
from termcolor import colored as clr

try:
    import sklearn.pipeline
    from sklearn.kernel_approximation import RBFSampler
except ModuleNotFoundError:
    print(
        clr(
            "Warning, for RadialBasisFunction feature extractor you need to"
            + " install sklearn.",
            "red",
            attrs=["bold"],
        )
    )


__all__ = ["Downsample", "Normalize", "RGB2Y", "Standardize", "RBFFeaturize"]


class AbstractTransformation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, o):
        pass

    def __str__(self):
        return "[{}]".format(type(self).__name__)


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
        return lycon.resize(
            o, width=self.width, height=self.height, interpolation=self.interpolation
        )


class RGB2Y(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.rgb = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

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
        env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float32
        )

    def transform(self, o):
        return np.array(o).astype(np.float32) / 255.0


class Standardize(AbstractTransformation):
    """ Scale to zero mean and unit variance """

    def __init__(self, samples):
        super().__init__()

        if isinstance(samples, list):
            samples = np.array(samples)
        self.mean = samples.mean(0)
        self.std = samples.std(0)
        assert self.mean.shape == self.std.shape == samples[0].shape
        print("stats: ", self.mean, self.std)

    def update_env_specs(self, env):
        shape = env.observation_space.low.shape
        env.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=shape, dtype=np.float32
        )

    def transform(self, o):
        return (o - self.mean) / self.std


class SmoothOneHot(object):
    pass


class RBFFeaturize(AbstractTransformation):
    """ Extract features using approximate RBF kernels.

        https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb
    """

    def __init__(self, samples, n_components=100):
        super().__init__()
        if isinstance(samples, list):
            samples = np.array(samples)

        # construct some RBFApproximators
        self.featurizer = sklearn.pipeline.FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components)),
            ]
        )
        self.shape = (n_components * 4,)

        # fit the approximators with the standardized data
        mu, std = (samples.mean(0), samples.std(0))
        std_samples = (samples - mu) / std
        self.featurizer.fit(std_samples)

    def update_env_specs(self, env):
        shape = env.observation_space.low.shape
        env.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=self.shape, dtype=np.float32
        )

    def transform(self, o):
        return self.featurizer.transform([o]).flatten()


def main():
    from wrappers import SqueezeRewards, TransformObservations

    env = gym.make("SpaceInvaders-v0")
    env = TransformObservations(env, [Downsample(84, 84), RGB2Y(), Normalize()])
    env = SqueezeRewards(env)

    o, done = env.reset(), False
    print(
        o.shape,
        o.dtype,
        "max=%3.2f" % o.max(),
        "min=%3.2f" % o.min(),
        "mean=%3.2f" % o.mean(),
    )
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


if __name__ == "__main__":
    main()
