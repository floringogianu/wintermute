""" Neural Network architecture for Atari games.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


__all__ = [
    "AtariNet",
    "BootstrappedAtariNet",
    "get_feature_extractor",
    "get_head",
]


def get_feature_extractor(input_depth):
    """ Configures the default Atari feature extractor. """
    return nn.Sequential(
        nn.Conv2d(input_depth, 32, kernel_size=8, stride=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
    )


def get_head(hidden_size, out_size, shared_bias=False):
    """ Configures the default Atari output layers. """
    if shared_bias:
        return nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_size),
            nn.ReLU(inplace=True),
            SharedBiasLinear(hidden_size, out_size),
        )
    return nn.Sequential(
        nn.Linear(64 * 7 * 7, hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, out_size),
    )


def init_weights(module):
    """ Callback for resetting a module's weights to Xavier Uniform and
        biases to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


def no_grad(module):
    """ Callback for turning off the gradient of a module.
    """
    try:
        module.weight.requires_grad = False
    except AttributeError:
        pass


class SharedBiasLinear(nn.Linear):
    """ Applies a linear transformation to the incoming data: `y = xA^T + b`.
        As opposed to the default Linear layer it has a shared bias term.
        This is employed for example in Double-DQN.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features, out_features):
        super(SharedBiasLinear, self).__init__(in_features, out_features, True)
        self.bias = Parameter(torch.Tensor(1))

    def extra_repr(self):
        return "in_features={}, out_features={}, bias=shared".format(
            self.in_features, self.out_features
        )


class AtariNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(  # pylint: disable=bad-continuation
        self, input_ch, hist_len, out_size, hidden_size=256, shared_bias=False
    ):
        super(AtariNet, self).__init__()

        self.__is_categorical = False
        if isinstance(out_size, tuple):
            self.__is_categorical = True
            self.__action_no, atoms_no = out_size
            out_size = self.__action_no * atoms_no

        self.__feature_extractor = get_feature_extractor(hist_len * input_ch)
        self.__head = get_head(hidden_size, out_size, shared_bias)

        self.reset_parameters()

    def forward(self, x):
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor"
        x = x.float().div_(255)

        x = self.__feature_extractor(x)
        x = x.view(x.size(0), -1)
        out = self.__head(x)

        if self.__is_categorical:
            splits = out.chunk(self.__action_no, 1)
            return torch.stack(list(map(lambda s: F.softmax(s), splits)), 1)
        return out

    def reset_parameters(self):
        """ Reinitializez parameters to Xavier Uniform for all layers and
            0 bias.
        """
        self.apply(init_weights)

    @property
    def feature_extractor(self):
        return self.__feature_extractor

    @property
    def head(self):
        return self.__head


class BootstrappedAtariNet(nn.Module):
    def __init__(self, proto, boot_no=10, full=False):
        """ Constructs a bootstrapped estimator using a prototype estimator.

            When `full` it simply duplicates and resets the weight
            initializaitons `boot_no` times.

            When not `full`, the ensemble is built by calling `head` and
            `feature_extractor` on the prototype estimator. It uses the
            `feature_extractor` as the common part of the ensemble and it
            duplicates the `head` component `boot_no` times.

        Args:
            proto (nn.Module): An estimator we ensemblify.
            boot_no (int, optional): Defaults to 10. Size of the ensemble.
            full (bool, optional): Defaults to False. When Trues we duplicate
            the full prototype. When False we only duplicate the `head` part
            of the ensemble.
        """
        super(BootstrappedAtariNet, self).__init__()

        self.__feature_extractor = None
        if full:
            self.__ensemble = [
                deepcopy(proto).reset_parameters() for _ in range(boot_no)
            ]
        else:
            try:
                self.__feature_extractor = deepcopy(proto.feature_extractor)
                self.__ensemble = [deepcopy(proto.head) for _ in range(boot_no)]
            except AttributeError as err:
                print(
                    "Your prototype model didn't implement `head` and "
                    + "`feature_extractor` getters. Either construct a full "
                    + "ensemble or implement them. Here's the rest of the error: ",
                    err,
                )

        self.__priors = [deepcopy(model) for model in self.__ensemble]
        for prior in self.__priors:
            prior.apply(no_grad)

        self.reset_parameters()

    def forward(self, x, mid=None):
        """ In training mode, when `mid` is provided, do an inference step
                through the ensemble component indicated by `mid`. Otherwise it
                returns the mean of the predictions of the ensemble.
            Args:
                x (torch.tensor): input of the model
                mid (int): id of the component in the ensemble to train on `x`.
            Returns:
                torch.tensor: the mean of the ensemble predictions.
        """
        if self.__feature_extractor is not None:
            x = self.__feature_extractor(x)
            x = x.view(x.size(0), -1)

        if mid is not None:
            y = self.__ensemble[mid](x)
            if self.__priors:
                y += self.__priors[mid](x)
            return y

        if self.__priors:
            ys = [m(x) + p(x) for m, p in zip(self.__ensemble, self.__priors)]
        else:
            ys = [model(x) for model in self.__ensemble]

        ys = torch.stack(ys, 0)

        return ys.mean(0), ys.var(0)

    def reset_parameters(self):
        self.apply(init_weights)
        for prior, model in zip(self.__priors, self.__ensemble):
            prior.apply(init_weights)
            model.apply(init_weights)

    def parameters(self, recurse=True):
        """ Groups the ensemble parameters so that the optimizer can keep
            separate statistics for each model in the ensemble.
        Returns:
            iterator: a group of parameters.
        """
        return [{"params": model.parameters()} for model in self.__ensemble]

    def __len__(self):
        return len(self.__ensemble)


if __name__ == "__main__":
    net = AtariNet(1, 4, 5)
    ens = BootstrappedAtariNet(net, 5)
    print(net)
    print(ens)

    print("\nSingle state.")
    print("q2: ", ens(torch.rand(1, 4, 84, 84), mid=2))
    print("q, σ: ", ens(torch.rand(1, 4, 84, 84)))

    print("\nBatch.")
    print("q2: ", ens(torch.rand(5, 4, 84, 84), mid=2))
    print("q, σ: ", ens(torch.rand(5, 4, 84, 84)))

    print("\nCheck param init:")
    head_params = net.head.parameters()
    print(f"proto weight:  ", next(head_params).data[0, :8])
    print(f"proto bias  :  ", next(head_params).data[:8])
    for i, p in enumerate(ens.parameters()):
        print(f"model{i} weight: ", next(p["params"]).data[0, :8])
        print(f"model{i} bias:   ", next(p["params"]).data[:8])
