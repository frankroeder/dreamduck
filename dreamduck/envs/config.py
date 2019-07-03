
from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'input_size',
                           'output_size', 'activation'])

games = {}

rnnenv = Game(env_name='rnnenv',
              input_size=576+512*1,
              output_size=1,
              activation='tanh',
              )
games['rnnenv'] = rnnenv

realenv = Game(env_name='realenv',
               input_size=576+512*1,
               output_size=1,
               activation='tanh',
               )
games['realenv'] = realenv
