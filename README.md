# Worldmodel DuckieTown

## Installation

1. Create virtualenv
2. Install dependcies `pip install -r ./dreamduck/envs/requirements.txt`
3. Install baseline with custom env import
  `pip install  git+https://github.com/Bassstring/baselines`
4. Install this module `pip install -e .`

## Training

Choose DreamDuck-v0 for the default environment. -v1 for the World Model
interpretation of the real environment and -v2 for the dream environment.

- `python -m baselines.run --alg=ppo2 --env=DreamDuck-v0 --num_timesteps=2e7
  --save_path=~/models/dreamduck__ppo2`

- `python -m baselines.run --alg=ppo2 --env=DreamDuck-v0 --num_timesteps=0
  --load_path=~/models/dreamduck__ppo2 --play`
# dreamduck
