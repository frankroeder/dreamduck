# World Model DuckieTown

This repository is an implementation of the [WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)
combined with forks of [gym-duckietown](https://github.com/Bassstring/gym-duckietown)
and [baselines](https://github.com/Bassstring/baselines).

There are three gym environment provided:

1. DreamDuck-v0 default environment `dreamduck/envs/env.py`
2. DreamDuck-v1 world model transformed environment `dreamduck/envs/realenv.py`
3. DreamDuck-v2 dream environment `dreamduck/envs/rnnenv.py`

To train the world model from scratch follow the introductions in
[this](https://github.com/Bassstring/dreamduck/blob/master/dreamduck/envs/README.md) readme.

## Getting Started

### Installation

1. Create a virtual environment with `python/python3 -m venv venv` and activate
  it with `source venv/bin/activate`
2. Install dependencies `pip install -r ./dreamduck/envs/requirements.txt`
3. Install baseline with custom environment import
  `pip install  git+https://github.com/Bassstring/baselines`
4. Install this module `pip install -e .`

## Manual Control

All three environment/representations are available to test out manually:

### Default environment

- `python dreamduck/envs/env.py`
- For help `python dreamduck/envs/env.py -h`

### World Model Interpretation of the real Environment

- `python dreamduck/envs/realenv.py`

### Dreaming without real Environment

- `python dreamduck/envs/rnnenv.py`

2. World Model Interpretation of the real environment `pytho dreamduck/envs/realenv.py`

### Training

Usage like `baselines===0.1.5`:

- `python -m baselines.run --alg=ppo2 --env=DreamDuck-v0 --num_timesteps=2e7 --num_env=4 --save_path=./models/dreamduck__ppo2`

- `python -m baselines.run --alg=ppo2 --env=DreamDuck-v0 --num_timesteps=0 --load_path=./models/dreamduck__ppo2 --play`

## Authors

Frank Röder
Shahd Safarani
