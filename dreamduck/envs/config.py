from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'input_size',
                           'output_size', 'activation'])

games = {}

duckietown = Game(env_name='duckietown',
                  input_size=576+512*1,
                  output_size=2,
                  activation='tanh',
                  )
games['duckietown'] = duckietown
