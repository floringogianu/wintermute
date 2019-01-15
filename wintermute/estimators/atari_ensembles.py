""" Here we implement an optimized version of an ensemble of Atari estimators
    and prove that we get the same result faster.

    The difference is that the natural way for a naive ensemble to output data
    is (heads_no x batch_size x actions_no), while for the flat ensemble yields
    outputs with shape (batch_size x heads-no x actions_no).
"""

import torch
from torch import nn
from torch.nn import functional as F


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


class AtariNet(nn.Module):
    """ This is the standard Atari convolutional neural network.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        hidden_size: int = 512,
        actions_no: int = 7,
        shared_bias: bool = True,
        hist_len: int = 4,
        **_kwargs,
    ) -> None:
        super().__init__()
        self._feature_extractor = get_feature_extractor(input_depth=hist_len)
        self._head = nn.Sequential(
            nn.Linear(64 * 49, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, actions_no)
            if not shared_bias
            else SharedBiasLinear(hidden_size, actions_no),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        if x.ndimension() == 4:
            x = self._feature_extractor(x)
            x = x.reshape(x.size(0), 64 * 49)
        elif x.ndimension() != 2:
            raise RuntimeError("Received a strange input.")
        return self._head(x)


class SharedBiasLinear(nn.Module):
    """ Linear layer with a single shared value for bias.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(1))

    def reset_parameters(self):
        """ Reinitializes weights using Xavier uniform (with zero bieas)
        """
        for module in self._feature_extractor:
            if hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight.data)
            if hasattr(module, "bias"):
                module.bias.data.zero_()
        for module in self._head:
            if hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight.data)
            if hasattr(module, "bias"):
                module.bias.data.zero_()

    def forward(self, x):  # pylint: disable=arguments-differ
        return F.linear(x, self.weight, self.bias)


class AtariEnsemble(nn.Module):
    """ The model to be used in our tests. Resembles AtariNet.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        heads_no: int = 10,
        hidden_size: int = 512,
        actions_no: int = 7,
        shared_bias: bool = True,
        hist_len: int = 4,
        **_kwargs,
    ) -> None:
        super().__init__()
        self._heads_no = heads_no = int(heads_no)
        self._hidden_size = hidden_size = int(hidden_size)
        self._shared_bias = shared_bias = bool(shared_bias)
        self._feature_extractor = get_feature_extractor(input_depth=hist_len)
        self._heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64 * 49, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_size, actions_no)
                    if not shared_bias
                    else SharedBiasLinear(hidden_size, actions_no),
                )
                for _ in range(heads_no)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitializes weights using Xavier uniform (with zero bieas)
        """
        for module in self._feature_extractor:
            if hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight.data)
            if hasattr(module, "bias"):
                module.bias.data.zero_()
        for head in self._heads:
            nn.init.xavier_uniform_(head[0].weight.data)
            head[0].bias.data.zero_()
            nn.init.xavier_uniform_(head[2].weight.data)
            head[2].bias.data.zero_()

    @property
    def heads_no(self) -> int:
        """ True if module uses threads when forwards through its heads.
        """
        return self._heads_no

    @property
    def hidden_size(self) -> int:
        """ True if module uses threads when forwards through its heads.
        """
        return self._hidden_size

    @property
    def shared_bias(self) -> bool:
        """ True if module uses threads when forwards through its heads.
        """
        return self._shared_bias

    def features(self, x):
        """ This function extracts the features (common to all components) for
            the given states.
        """
        x = self._feature_extractor(x)
        return x.reshape(x.size(0), 64 * 49)

    def forward(self, x, head_idx=None):  # pylint: disable=arguments-differ
        """ Returns either batch_size x actions_no if head_idx is given,
            or heads_no x batch_size x actions_no if head_idx is None
        """
        if x.ndimension() == 4:
            x = self._feature_extractor(x)
            x = x.reshape(x.size(0), 64 * 49)
        elif x.ndimension() != 2:
            raise RuntimeError("Received a strange input.")
        if head_idx is None:
            output = torch.stack(tuple(head(x) for head in self._heads))
        else:
            output = self._heads[head_idx](x)
        return output


class FlatAtariEnsemble(nn.Module):
    """ The flat version of the above module.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        heads_no: int = 10,
        hidden_size: int = 512,
        actions_no: int = 10,
        shared_bias: bool = True,
        hist_len: int = 4,
        transpose_head_weights: bool = False,
    ) -> None:
        super().__init__()
        self._heads_no = heads_no = int(heads_no)
        self._hidden_size = hidden_size = int(hidden_size)
        self._actions_no = actions_no = int(actions_no)

        self._transpose_head_weights = transpose = bool(transpose_head_weights)

        self._feature_extractor = get_feature_extractor(input_depth=hist_len)
        self._flat_hidden = nn.Linear(64 * 49, hidden_size * heads_no)
        self._head_weights = nn.Parameter(
            torch.randn(heads_no, actions_no, hidden_size)
            if transpose
            else torch.randn(heads_no, hidden_size, actions_no)
        )
        self._shared_bias = shared_bias
        if shared_bias:
            self._head_bias = nn.Parameter(torch.randn(heads_no, 1))
        else:
            self._head_bias = nn.Parameter(torch.randn(heads_no, actions_no))

        self.__reset_parameters()

    def reset_parameters(self):
        """ Reinitializes weights using Xavier uniform (with zero bieas)
        """
        self.__reset_parameters()

    def __reset_parameters(self):
        """ Reinitializes weights using Xavier uniform (with zero bieas)
        """
        for module in self._feature_extractor:
            if hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight.data)
            if hasattr(module, "bias"):
                module.bias.data.zero_()
        hidden_sz = self._hidden_size
        for head_idx in range(self.heads_no):
            pre_slice = slice(hidden_sz * head_idx, (head_idx + 1) * head_idx)
            nn.init.xavier_uniform_(self._flat_hidden.weight[pre_slice].data)
            if self._transpose_head_weights:
                nn.init.xavier_uniform_(self._head_weights[head_idx].data)
            else:
                nn.init.xavier_uniform_(
                    self._head_weights[head_idx].data.transpose(0, 1)
                )

        self._flat_hidden.bias.data.zero_()
        self._head_bias.data.zero_()

    @property
    def heads_no(self) -> int:
        """ True if module uses threads when forwards through its heads.
        """
        return self._heads_no

    @property
    def hidden_size(self) -> int:
        """ True if module uses threads when forwards through its heads.
        """
        return self._hidden_size

    @property
    def shared_bias(self) -> bool:
        """ True if module uses threads when forwards through its heads.
        """
        return self._shared_bias

    @property
    def head_weights_transposed(self) -> bool:
        """ It seems that bmm and matmul are faster if the second matrix is
            not transposed. So we support this.
        """
        return self._transpose_head_weights

    def forward(self, x, head_idx=None):  # pylint: disable=arguments-differ

        if x.ndimension() == 4:
            x = self._feature_extractor(x)
            x = x.reshape(x.size(0), -1)
        elif x.ndimension() != 2:
            raise RuntimeError("Received a strange input.")

        batch_sz = x.size(0)
        hidden_sz = self._hidden_size
        heads_no = self._heads_no
        head_weights, head_bias = self._head_weights, self._head_bias

        if head_idx is None:
            pre_outputs = F.relu(self._flat_hidden(x), inplace=True)
            return torch.baddbmm(
                head_bias.repeat(batch_sz, 1).unsqueeze(1),
                pre_outputs.view(batch_sz * heads_no, 1, hidden_sz),
                head_weights.repeat(batch_sz, 1, 1).transpose(1, 2)
                if self._transpose_head_weights
                else head_weights.repeat(batch_sz, 1, 1),
            ).view(batch_sz, heads_no, self._actions_no)
            """
            return (
                torch.bmm(
                    pre_outputs.view(batch_sz * heads_no, 1, hidden_sz),
                    head_weights.repeat(batch_sz, 1, 1).transpose(1, 2),
                ).view(batch_sz, heads_no, self._actions_no)
                + head_bias
            )
            return (
                torch.matmul(
                    pre_outputs.view(batch_sz, heads_no, 1, hidden_sz),
                    head_weights.unsqueeze(0) #.transpose(2, 3),
                ).squeeze(2)
                + head_bias
            )
            """

        else:
            pre_slice = slice(head_idx * hidden_sz, (head_idx + 1) * hidden_sz)
            pre_weight = self._flat_hidden.weight[pre_slice]
            pre_bias = self._flat_hidden.bias[pre_slice]
            hidden = F.relu(F.linear(x, weight=pre_weight, bias=pre_bias))

            head_weight = (
                head_weights[head_idx]
                if self._transpose_head_weights
                else head_weights[head_idx].transpose(0, 1)
            )

            return F.linear(
                hidden, weight=head_weight, bias=head_bias[head_idx]
            )


class FlatAtariEnsembleWithPriors(FlatAtariEnsemble):
    """ Ensemble with priors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.priors = [FlatAtariEnsemble(**kwargs)]

    def reset_parameters(self):
        """ Reinitializes weights using Xavier uniform (with zero bieas)
        """
        super().reset_parameters()
        self.priors[0].reset_parameters()

    def to(self, device):  # pylint: disable=arguments-differ
        super().to(device)
        self.priors[0].to(device)

    def forward(self, x, head_idx=None):  # pylint: disable=arguments-differ
        output = super().forward(x, head_idx=head_idx)
        with torch.no_grad():
            prior = self.priors[0](x, head_idx=head_idx)
        return output + prior

    def state_dict(self, **kwargs):  # pylint: disable=arguments-differ
        return (
            super().state_dict(**kwargs),
            self.priors[0].state_dict(**kwargs),
        )

    def load_state_dict(  # pylint: disable=arguments-differ, bad-continuation
        self, state_dict, **kwargs
    ):
        ensemble_state, prior_state = state_dict
        super().load_state_dict(ensemble_state, **kwargs)
        self.priors[0].load_state_dict(prior_state, **kwargs)


def copy_naive_to_flat(source: AtariEnsemble, dest: FlatAtariEnsemble):
    """ In here we copy parameters from a naive AtariEnsemble to a
        FlatAtariEnsemble.
    """
    # pylint: disable=protected-access
    if (  # pylint: disable=bad-continuation
        (source.shared_bias != dest.shared_bias)
        or (source.heads_no != dest.heads_no)
        or (source.hidden_size != dest.hidden_size)
    ):
        raise RuntimeError("Ensembles should have same configuration")

    with torch.no_grad():
        dest._feature_extractor.load_state_dict(
            source._feature_extractor.state_dict()
        )

        for k in range(source.heads_no):
            idxs = slice(k * source.hidden_size, (k + 1) * source.hidden_size)

            dest._flat_hidden.weight.data[idxs].copy_(
                source._heads[k][0].weight.data
            )
            dest._flat_hidden.bias.data[idxs].copy_(
                source._heads[k][0].bias.data
            )
            if dest.head_weights_transposed:
                dest._head_weights.data[k].copy_(
                    source._heads[k][2].weight.data
                )
            else:
                dest._head_weights.data[k].copy_(
                    source._heads[k][2].weight.data.transpose(0, 1)
                )

            dest._head_bias[k].data.copy_(source._heads[k][2].bias.data)


def copy_flat_to_naive(source: FlatAtariEnsemble, dest: AtariEnsemble):
    """ In here we copy parameters from a naive AtariEnsemble to a
        FlatAtariEnsemble.
    """
    # pylint: disable=protected-access
    if (  # pylint: disable=bad-continuation
        (source.shared_bias != dest.shared_bias)
        or (source.heads_no != dest.heads_no)
        or (source.hidden_size != dest.hidden_size)
    ):
        raise RuntimeError("Ensembles should have same configuration")

    with torch.no_grad():
        dest._feature_extractor.load_state_dict(
            source._feature_extractor.state_dict()
        )

        for k in range(source.heads_no):
            idxs = slice(k * source.hidden_size, (k + 1) * source.hidden_size)

            dest._heads[k][0].weight.data.copy_(
                source._flat_hidden.weight.data[idxs]
            )
            dest._heads[k][0].bias.data.copy_(
                source._flat_hidden.bias.data[idxs]
            )
            if source.head_weights_transposed:
                dest._heads[k][2].weight.data.copy_(
                    source._head_weights.data[k]
                )
            else:
                dest._heads[k][2].weight.data.copy_(
                    source._head_weights.data[k].transpose(0, 1)
                )

            dest._heads[k][2].bias.data.copy_(source._head_bias[k].data)


if __name__ == "__main__":
    from time import time
    from termcolor import colored as clr
    from torch import optim  # pylint: disable=ungrouped-imports

    def __starting(msg, kwargs):
        print(
            clr("Starting!", "yellow"),
            msg,
            "(",
            ",".join([f"{k}={v}" for (k, v) in kwargs.items()]),
            ")",
        )

    def __success(msg, kwargs):
        print(
            clr("OK!", "yellow"),
            msg,
            "(",
            ",".join([f"{k}={v}" for (k, v) in kwargs.items()]),
            ")",
        )

    def __naive_to_flat_test(atol=1e-07, **kwargs):
        """ This functions checks that copying parameters from a naive ensemble
            to a flat one works and also proves they represent the same
            function.
        """

        __starting("Copying from naive to flat.", kwargs)

        naive_model = AtariEnsemble(**kwargs)
        flat_model = FlatAtariEnsemble(**kwargs)
        naive_model.to("cuda")
        flat_model.to("cuda")
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")

        with torch.no_grad():
            before_naive = naive_model(eval_state)
            before_flat = flat_model(eval_state)
        copy_naive_to_flat(naive_model, flat_model)
        with torch.no_grad():
            after_naive = naive_model(eval_state)
            after_flat = flat_model(eval_state)

        assert torch.allclose(before_naive, after_naive, atol=atol)
        assert torch.allclose(
            after_naive.transpose(0, 1), after_flat, atol=atol
        )
        assert not torch.allclose(before_flat, after_flat, atol=atol)

        for head_idx in range(kwargs["heads_no"]):
            eval_state = torch.randn(4, 4, 84, 84).to("cuda")
            flat_output = flat_model(eval_state, head_idx)
            naive_output = naive_model(eval_state, head_idx)
            assert torch.allclose(flat_output, naive_output, atol=atol)

        __success("Copying from naive to flat.", kwargs)

    def __flat_to_naive_test(atol=1e-07, **kwargs):
        """ This functions checks that copying parameters from a flat ensemble
            to a naive one works and also proves they represent the same
            function.
        """

        __starting("Copying from flat to naive.", kwargs)

        naive_model = AtariEnsemble(**kwargs)
        flat_model = FlatAtariEnsemble(**kwargs)
        naive_model.to("cuda")
        flat_model.to("cuda")
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")

        with torch.no_grad():
            before_naive = naive_model(eval_state)
            before_flat = flat_model(eval_state)
        copy_flat_to_naive(flat_model, naive_model)
        with torch.no_grad():
            after_naive = naive_model(eval_state)
            after_flat = flat_model(eval_state)

        assert torch.allclose(
            before_flat, after_flat, atol=atol
        ), f"err={(before_flat - after_flat).abs().max().item()}"
        assert torch.allclose(
            after_naive.transpose(0, 1), after_flat, atol=atol
        )
        assert not torch.allclose(before_naive, after_naive, atol=atol)

        for head_idx in range(kwargs["heads_no"]):
            eval_state = torch.randn(4, 4, 84, 84).to("cuda")
            flat_output = flat_model(eval_state, head_idx)
            naive_output = naive_model(eval_state, head_idx)
            assert torch.allclose(flat_output, naive_output, atol=atol)

        # Test the mask

        state = torch.randn(32, 4, 84, 84).to("cuda")
        mask = (
            torch.bernoulli(torch.rand(32, kwargs["heads_no"]))
            .byte()
            .to("cuda")
        )

        flat_output = flat_model(state)[mask]
        features = naive_model.features(state)
        naive_outputs = []
        for head_idx, batch_mask in enumerate(mask.transpose(0, 1)):
            naive_outputs.append(
                naive_model(features[batch_mask], head_idx=head_idx)
            )

        naive_output = torch.cat(naive_outputs, dim=0)
        assert torch.allclose(
            flat_output.sum(dim=0), naive_output.sum(dim=0), atol=atol
        )

        __success("Copying from flat to naive.", kwargs)

    def __optimizer_goes_hand_in_hand(atol=1e-07, **kwargs):
        """ This function checks that performing a few optimization steps (SGD
            with momentum) on both ensembles yields (almost) identical results.
        """

        __starting("Optimizing both.", kwargs)

        naive_model = AtariEnsemble(**kwargs)
        flat_model = FlatAtariEnsemble(**kwargs)
        naive_model.to("cuda")
        flat_model.to("cuda")
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")

        states = torch.randn(32, 4, 84, 84).to("cuda")
        targets = torch.randn(32, kwargs["heads_no"], kwargs["actions_no"]).to(
            "cuda"
        )
        copy_naive_to_flat(naive_model, flat_model)

        with torch.no_grad():
            before_naive = naive_model(eval_state)
            before_flat = flat_model(eval_state)

        assert torch.allclose(
            before_naive.transpose(0, 1), before_flat, atol=atol
        )

        mse = nn.MSELoss(reduction="mean")

        naive_optimizer = optim.SGD(
            naive_model.parameters(), lr=1e-3, momentum=0.90
        )
        flat_optimizer = optim.SGD(
            flat_model.parameters(), lr=1e-3, momentum=0.90
        )

        for _ in range(10):
            naive_optimizer.zero_grad()
            mse(naive_model(states), targets.transpose(0, 1)).backward()
            naive_optimizer.step()

            flat_optimizer.zero_grad()
            mse(flat_model(states), targets).backward()
            flat_optimizer.step()

        after_naive = naive_model(eval_state)
        after_flat = flat_model(eval_state)

        assert torch.allclose(
            after_naive.transpose(0, 1), after_flat, atol=atol
        )

        __success("Optimizing both.", kwargs)

    def __test_priors(atol=1e-07, **kwargs):
        __starting("Using priors.", kwargs)

        ensemble = FlatAtariEnsembleWithPriors(**kwargs)
        ensemble.to("cuda")
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")

        with torch.no_grad():
            before_restore = ensemble(eval_state)

        state_dict = ensemble.state_dict()

        assert isinstance(state_dict, tuple) and len(state_dict) == 2

        del ensemble

        new_ensemble = FlatAtariEnsembleWithPriors(**kwargs)
        new_ensemble.to("cuda")
        new_ensemble.load_state_dict(state_dict)
        with torch.no_grad():
            after_restore = new_ensemble(eval_state)

        assert torch.allclose(before_restore, after_restore, atol=atol)

        __success("Using priors.", kwargs)

    # -- Below are benchmarks

    def __report_speedup(  # pylint: disable=bad-continuation
        msg,
        new_time,
        old_time,
        new_name="flat ensemble",
        old_name="serial ensemble",
    ):
        speed_up = 100.0 * (old_time - new_time) / old_time
        print(
            msg
            + " ("
            + clr(f"{new_name}", "yellow")
            + " vs "
            + clr(f"{old_name}", "blue")
            + "): "
            + clr(f"{new_time:.2f}s.", "yellow")
            + " - "
            + clr(f"{old_time:.2f}s.", "blue")
            + f" | speed-up: "
            + clr(
                f"{speed_up:.2f}%",
                "green" if speed_up > 0 else "red",
                attrs=["bold"],
            )
        )

    def __flat_vs_naive_masked(flat_model, naive_model, nepochs, data, masks):
        """ Here we compare optimization times when using masks with naive, and
            flat ensembles..
        """

        assert isinstance(flat_model, FlatAtariEnsemble)
        assert isinstance(naive_model, AtariEnsemble)

        mse = nn.MSELoss(reduction="mean")

        flat_optimizer = optim.Adam(flat_model.parameters(), lr=1e-3)
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")
        with torch.no_grad():
            flat_model(eval_state)
        torch.cuda.synchronize()

        start = time()
        for _epoch in range(nepochs):
            for mask, (states, targets) in zip(masks, data):
                flat_optimizer.zero_grad()
                mse(flat_model(states)[mask], targets[mask]).backward()
                flat_optimizer.step()
        end = time()
        flat_train_time = end - start

        naive_optimizer = optim.Adam(naive_model.parameters(), lr=1e-3)
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")
        with torch.no_grad():
            naive_model(eval_state)

        start = time()
        for _epoch in range(nepochs):
            for mask, (states, targets) in zip(masks, data):
                naive_optimizer.zero_grad()
                features = naive_model.features(states)
                loss = 0
                for head_idx, batch_mask in enumerate(mask.transpose(0, 1)):
                    loss += mse(
                        naive_model(features[batch_mask], head_idx=head_idx),
                        targets[batch_mask, head_idx],
                    )
                loss.backward()
                naive_optimizer.step()
        end = time()
        naive_train_time = end - start

        __report_speedup("TRAINING (p=.5)", flat_train_time, naive_train_time)

    def __flat_vs_naive_full(flat_model, naive_model, nepochs, data):
        assert isinstance(flat_model, FlatAtariEnsemble)
        assert isinstance(naive_model, AtariEnsemble)

        mse = nn.MSELoss(reduction="mean")

        flat_optimizer = optim.Adam(flat_model.parameters(), lr=1e-3)
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")
        with torch.no_grad():
            flat_model(eval_state)
        torch.cuda.synchronize()

        start = time()
        for _epoch in range(nepochs):
            for states, targets in data:
                flat_optimizer.zero_grad()
                mse(flat_model(states), targets).backward()
                flat_optimizer.step()
        end = time()
        flat_train_time = end - start

        naive_optimizer = optim.Adam(naive_model.parameters(), lr=1e-3)
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")
        with torch.no_grad():
            naive_model(eval_state)
        torch.cuda.synchronize()

        start = time()
        for _epoch in range(nepochs):
            for states, targets in data:
                naive_optimizer.zero_grad()
                mse(naive_model(states), targets.transpose(0, 1)).backward()
                naive_optimizer.step()
        end = time()
        naive_train_time = end - start

        __report_speedup("FULL TRAINING", flat_train_time, naive_train_time)

    def __flat_vs_naive_inference(flat_model, naive_model, nepochs, data):
        assert isinstance(flat_model, FlatAtariEnsemble)
        assert isinstance(naive_model, AtariEnsemble)

        _val = 0
        start = time()
        with torch.no_grad():
            for _ in range(nepochs):
                for step, (state, _) in enumerate(data):
                    head_idx = (step // 50) % flat_model.heads_no
                    _val += flat_model(state, head_idx=head_idx).mean().item()
        end = time()
        flat_infer_time = end - start

        _val = 0
        start = time()
        with torch.no_grad():
            for _ in range(nepochs):
                for step, (state, _) in enumerate(data):
                    head_idx = (step // 50) % flat_model.heads_no
                    _val += naive_model(state, head_idx=head_idx).mean().item()
        end = time()
        naive_infer_time = end - start

        __report_speedup("INFERENCE", flat_infer_time, naive_infer_time)

    def __flat_vs_naive(ndata=100, nepochs=10, **kwargs):
        naive_model = AtariEnsemble(**kwargs)
        flat_model = FlatAtariEnsemble(**kwargs)
        naive_model.to("cuda")
        flat_model.to("cuda")

        data = [
            (
                torch.randn(32, 4, 84, 84).to("cuda"),
                torch.randn(32, kwargs["heads_no"], kwargs["actions_no"]).to(
                    "cuda"
                ),
            )
            for _ in range(ndata)
        ]
        masks = [
            torch.bernoulli(torch.rand(32, kwargs["heads_no"]))
            .byte()
            .to("cuda")
            for _ in range(ndata)
        ]

        print(
            f"Will perform {ndata * nepochs:d} steps for training, and "
            f"{ndata * nepochs * 10:d} steps for inference; "
        )

        __flat_vs_naive_masked(flat_model, naive_model, nepochs, data, masks)
        __flat_vs_naive_full(flat_model, naive_model, nepochs, data)
        __flat_vs_naive_inference(flat_model, naive_model, nepochs * 10, data)

    def __flat_vs_atari_full(ensemble, atari_net, nepochs, data):
        assert isinstance(ensemble, FlatAtariEnsemble)
        assert isinstance(atari_net, AtariNet)
        mse = nn.MSELoss(reduction="mean")

        ensemble_optimizer = optim.Adam(ensemble.parameters(), lr=1e-3)
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")
        with torch.no_grad():
            ensemble(eval_state)
        torch.cuda.synchronize()

        start = time()
        for _epoch in range(nepochs):
            for states, targets in data:
                ensemble_optimizer.zero_grad()
                mse(ensemble(states).mean(dim=1), targets).backward()
                ensemble_optimizer.step()
        end = time()
        ensemble_train_time = end - start

        atari_net_optimizer = optim.Adam(atari_net.parameters(), lr=1e-3)
        eval_state = torch.randn(4, 4, 84, 84).to("cuda")
        with torch.no_grad():
            atari_net(eval_state)
        torch.cuda.synchronize()

        start = time()
        for _epoch in range(nepochs):
            for states, targets in data:
                atari_net_optimizer.zero_grad()
                mse(atari_net(states), targets).backward()
                atari_net_optimizer.step()
        end = time()
        atari_net_train_time = end - start
        __report_speedup(
            "FULL TRAINING",
            ensemble_train_time,
            atari_net_train_time,
            new_name="flat ensemble",
            old_name="atari-net",
        )

    def __flat_vs_atari_inference(ensemble, atari_net, nepochs, data):
        assert isinstance(ensemble, FlatAtariEnsemble)
        assert isinstance(atari_net, AtariNet)

        _value = 0
        start = time()
        with torch.no_grad():
            for _ in range(nepochs):
                for (state, _) in data:
                    _value += ensemble(state).mean().item()
        end = time()
        ensemble_infer_time = end - start

        _value = 0
        start = time()
        with torch.no_grad():
            for _ in range(nepochs):
                for (state, _) in data:
                    _value += atari_net(state).mean().item()
        end = time()
        atari_net_infer_time = end - start

        __report_speedup(
            "INFERENCE",
            ensemble_infer_time,
            atari_net_infer_time,
            new_name="flat ensemble",
            old_name="atari-net",
        )

    def __flat_vs_atari(ndata=100, nepochs=10, **kwargs):
        atari_net = AtariNet(**kwargs)
        ensemble = FlatAtariEnsembleWithPriors(**kwargs)
        atari_net.to("cuda")
        ensemble.to("cuda")

        data = [
            (
                torch.randn(32, 4, 84, 84).to("cuda"),
                torch.randn(32, kwargs["actions_no"]).to("cuda"),
            )
            for _ in range(ndata)
        ]

        print(
            f"Will perform {ndata * nepochs:d} steps for training, and "
            f"{ndata * nepochs * 10:d} steps for inference; "
        )

        __flat_vs_atari_full(ensemble, atari_net, nepochs, data)
        __flat_vs_atari_inference(ensemble, atari_net, nepochs * 10, data)

    def main():
        """ Here we test the above cool modules.
        """
        atol = 1e-05  # decrease this if problematic
        kwargs = {
            "actions_no": 18,
            "hidden_size": 512,
            "heads_no": 10,
            "shared_bias": True,
        }
        ndata = 50
        nepochs = 10

        for benchmark in [__flat_vs_atari, __flat_vs_naive]:
            print("\nBenchmark:", benchmark.__name__)
            for shared_bias in [True, False]:
                for hwt in [True, False]:
                    kwargs["shared_bias"] = shared_bias
                    kwargs["transpose_head_weights"] = hwt
                    print("")
                    print(",".join([f"{k}={v}" for (k, v) in kwargs.items()]))
                    benchmark(ndata=ndata, nepochs=nepochs, **kwargs)

        print("\nBENCHMARKS OVER\n")

        tests = [
            __naive_to_flat_test,
            __flat_to_naive_test,
            __optimizer_goes_hand_in_hand,
            __test_priors,
        ]
        for test_func in tests:
            print("\nTest:", test_func.__name__)
            for shared_bias in [True, False]:
                for hwt in [True, False]:
                    kwargs["shared_bias"] = shared_bias
                    kwargs["transpose_head_weights"] = hwt
                    test_func(atol=atol, **kwargs)

        print("\nTESTS PASSED\n")

    main()
