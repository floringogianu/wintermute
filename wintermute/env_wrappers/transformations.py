""" Transformations that work with `Transform Observation`.

Example:

.. code-block:: python

    env = TransformObservations(
        env,
        [
            T.Downsample(84, 84),
            T.RGB2Y()
            T.Normalize()
        ],
    )
"""
from abc import ABC, abstractmethod

import gym
import numpy as np
from termcolor import colored as clr

try:
    import cv2  #pylint: disable=import-error
except ModuleNotFoundError as err:
    print(
        clr(
            "OpenCV is required when using the Downsample wrapper. "
            + "Try `conda install -c menpo opencv`."
        ),
        err,
    )

try:
    import sklearn.pipeline  #pylint: disable=import-error
    from sklearn.kernel_approximation import RBFSampler  #pylint: disable=import-error
except ModuleNotFoundError as err:
    print(
        clr(
            "Warning, for RadialBasisFunction feature extractor you need to"
            + " install sklearn.",
            "red",
            attrs=["bold"],
        ),
        err,
    )


__all__ = ["Downsample", "Normalize", "RGB2Y", "Standardize", "RBFFeaturize"]


class AbstractTransformation(ABC):
    """Interface for `Transformation` objects.
    """

    @abstractmethod
    def transform(self, obs):
        """ Overwritten by each Transformation.
        """

    @abstractmethod
    def update_env_specs(self, env):
        """ Called by :class:`wintermute.wrappers.TransformObservation` to update
        environment specification.
        """

    def __str__(self):
        return "[{}]".format(type(self).__name__)


class Downsample(AbstractTransformation):
    """ Downsamples an RGB image using `lcon`.

    Args:
        height (int): Target height
        width (int): Target width
        interpolation ([type], optional): Defaults to linear. Interpolation
            algorithms. These are class members.

    Available interpolations are:
    """

    nearest = cv2.INTER_NEAREST
    linear = cv2.INTER_LINEAR
    cubic = cv2.INTER_CUBIC
    area = cv2.INTER_AREA
    lanczos = cv2.INTER_LANCZOS4

    def __init__(self, height, width, interpolation=linear):
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation = interpolation
        if interpolation is Downsample.lanczos:
            print("Warning, Lanczos interpolation can be slow.")

    def update_env_specs(self, env):
        obs_space = env.observation_space
        low, high, dtype = obs_space.low, obs_space.high, obs_space.dtype
        low = np.resize(low, (self.height, self.width, 3))
        high = np.resize(high, (self.height, self.width, 3))
        env.observation_space = gym.spaces.Box(low, high, dtype=dtype)

    def transform(self, obs):
        return cv2.resize(
            obs, (self.width, self.height), interpolation=self.interpolation
        )


class RGB2Y(AbstractTransformation):
    """ RGB to Luminance transformation.
    """
    def __init__(self):
        super().__init__()
        self.rgb = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    def update_env_specs(self, env):
        obs_space = env.observation_space
        low, high, dtype = obs_space.low, obs_space.high, obs_space.dtype
        shape = (low.shape[0], low.shape[1], 1)
        low, high = np.resize(low, shape), np.resize(high, shape)
        env.observation_space = gym.spaces.Box(low, high, dtype=dtype)

    def transform(self, obs):
        dtype = obs.dtype
        shape = (obs.shape[0], obs.shape[1], 1)
        return np.dot(obs, self.rgb).reshape(shape).astype(dtype)


class Normalize(AbstractTransformation):
    def __init__(self):
        super().__init__()

    def update_env_specs(self, env):
        shape = env.observation_space.low.shape
        env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float32
        )

    def transform(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class Standardize(AbstractTransformation):
    """ Scale to zero mean and unit variance. """

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

    def transform(self, obs):
        return (obs - self.mean) / self.std


class SmoothOneHot(object):
    pass


class RBFFeaturize(AbstractTransformation):
    """ Extract features using approximate RBF kernels.

    Lifted from `Denny Britz notebook <https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb>`_.
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

    def transform(self, obs):
        return self.featurizer.transform([obs]).flatten()


def main():
    from .wrappers import SqueezeRewards, TransformObservations

    env = gym.make("SpaceInvaders-v0")
    env = TransformObservations(env, [Downsample(84, 84), RGB2Y(), Normalize()])
    env = SqueezeRewards(env)

    obs, done = env.reset(), False
    print(
        obs.shape,
        obs.dtype,
        "max=%3.2f" % obs.max(),
        "min=%3.2f" % obs.min(),
        "mean=%3.2f" % obs.mean(),
    )
    print(env.observation_space, env.unwrapped.observation_space)
    print(env)


if __name__ == "__main__":
    main()
