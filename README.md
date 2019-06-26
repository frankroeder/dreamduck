# World Model DuckieTown

The actual world model is packaged as a model with the name dreamduck.
Different gym environments are installed and registered to be used system
wide the virtual environment. There are 3 types of envs supplied by the module.

1. DreamDuck-v0 default environment `dreamduck/envs/env.py`
2. DreamDuck-v1 world model transformed env `dreamduck/envs/realenv.py`
3. DreamDuck-v2 dream environment `dreamduck/envs/rnnenv.py`

To have capabilities the frame work baselines,
basic commands to train and test the learned model are possible to use.

Please run baselines from the root of this git repository.

## Installation

1. Create virtualenv
2. Install dependencies `pip install -r ./dreamduck/envs/requirements.txt`
3. Install baseline with custom env import
  `pip install  git+https://github.com/Bassstring/baselines`
4. Install this module `pip install -e .`

## Training

Choose DreamDuck-v0 for the default environment. -v1 for the World Model
interpretation of the real environment and -v2 for the dream environment.

- `python -m baselines.run --alg=ppo2 --env=DreamDuck-v0 --num_timesteps=2e7
  --num_env=4 --save_path=./models/dreamduck__ppo2`

- `python -m baselines.run --alg=ppo2 --env=DreamDuck-v0 --num_timesteps=0
  --load_path=./models/dreamduck__ppo2 --play`
