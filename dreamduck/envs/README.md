# Dreamduck

_Dreamduck_ is a research project to investigate the approach of hallucination
based learning in the environment of _Duckietown_.

## Installation

`pip install -r requirements.txt`

## Usage

### Manual

`python env.py --map-name loop_pedestrians`
`python env.py --map-name loop_dyn_duckiebots `

## Help

`python env.py -h`

## Training

### Generate Training Examples

Use python `extract.py` to generate rollouts records or
`python extract.py --debug --full-episode` for debugging and full episode.

## TODOS

- cleanup code

## Pipeline

1. Generate 10000 rollouts with `extract.bash`
  - This script will start `extract.py`
  - The default configuration will start X processes recording the agent
    with its random policy actions in the wrapped environment
  - Each process/worker will execute `MAX_TRIALS` of rollouts
  - A rollout is limited to `Max_FRAMES` (only important if `full_episode=True`)
  - An `.npz` file is written to the folder `records/`containing the action
    and observation for each frame
2. Train the VAE to represent an observation just with `z`
  - Setup GPU and train the VAE
  - Manually validate the result with the jupyter notebook
3. Create series with `series.py` of VAE-encoded rollouts
  - Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod
  - Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod
4. Train the MDN-RNN
  - Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod
  - Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod
5. Train the agent
  - Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod
  - Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod
