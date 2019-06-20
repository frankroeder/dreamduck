import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DreamDuck-v0',
    entry_point='dreamduck.envs:DuckieTownWrapper',
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='DreamDuck-v1',
    entry_point='dreamduck.envs:DuckieTownReal',
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='DreamDuck-v2',
    entry_point='dreamduck.envs:DuckieTownRNN',
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)
