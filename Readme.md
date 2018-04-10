# Wintermute

**Work in progress. API will change frequently in the next weeks.**
Wintermute is a small and pragmatic library for Reinforcement Learning primitives. 
It will eventually contain, in a modular fashion, all the wrappers, utilities and 
building blocks necessary for developing new RL methods in an efficient manner. 
The aim is to be able to quickly compose new algorithms using well-tested functions 
and modules from this library.

## Installation

This installation assumes a `conda` environment. It was tested on `ubuntu`
only.

First we need to install some system dependencies required by `gym` and
`lycon`. You can find more about that on their respective GitHub project pages.

```bash
sudo apt-get install build-essential cmake libjpeg-dev libpng-dev zlib1g-dev
xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

Next you can simply create a conda environment containing wintermute and all
its dependencies:
```bash
conda env create -f wintermute.yml
```

Supposing you have most of the python dependencies such as `gym`, `pytorch` and `lycon`
and you don't want to install a separate `conda env`, you can also directly install 
`wintermute` with: 
```bash 
pip install git+git://github.com/floringogianu/wintermute.git --process-dependency-links
```

## To Do

- [ ] Add policy gradient primitives.
- [ ] Add an actor critic example in order to further figure out the direction
  of the API.
- [ ] Expose some design principles as soon as possible.


## Credits

- [@tudor-berariu](https://github.com/tudor-berariu) for some of the generators
  in
  [policy_evaluation/exploration_schedules.py](https://github.com/floringogianu/wintermute/blob/master/policy_evaluation/exploration_schedules.py).
- I took some of the wrappers from [OpenAI-Baselines](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py).
