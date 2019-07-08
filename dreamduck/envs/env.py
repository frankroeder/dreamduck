import numpy as np
import sys
import pyglet
from pyglet.window import key
import argparse
from gym.spaces.box import Box
from gym_duckietown.envs import DuckietownEnv
from cv2 import resize

SCREEN_X = 64
SCREEN_Y = 64


def _process_frame(frame):
    obs = frame[:, :, :].astype(np.float)/255.0
    obs = np.array(resize(obs, (SCREEN_X, SCREEN_Y)))
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    return obs


class DuckieTownWrapper(DuckietownEnv):
    def __init__(self, full_episode=False):
        super(DuckieTownWrapper, self).__init__(
            camera_width=SCREEN_X,
            camera_height=SCREEN_Y,
            map_name='loop_dyn_duckiebots',
            domain_rand=False,
            seed=0
        )
        self.full_episode = full_episode
        self.observation_space = Box(
            low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3))

    def _step(self, action):
        obs, reward, done, _ = super(DuckieTownWrapper, self).step(action)
        if self.full_episode:
            return _process_frame(obs), reward, False, {}
        return _process_frame(obs), reward, done, {}

    def _render(self, mode='human'):
        self.render()

    def _reset(self):
        self.reset()


def make_env(env_name, seed=-1, render_mode=True, load_model=True,
             full_episode=True):
    if env_name == 'default':
        env = DuckieTownWrapper(full_episode=full_episode)
    elif env_name == 'rnnenv':
        from rnnenv import DuckieTownRNN
        env = DuckieTownRNN(render_mode=render_mode, load_model=load_model)
    elif env_name == 'realenv':
        from realenv import DuckieTownReal
        env = DuckieTownReal(render_mode=render_mode, load_model=load_model)
    if seed >= 0:
        env.seed(seed)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', default='loop_dyn_duckiebots')
    parser.add_argument('--distortion', default=False, action='store_true')
    parser.add_argument('--draw-curve', action='store_true',
                        help='draw the lane following curve')
    parser.add_argument('--draw-bbox', action='store_true',
                        help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', action='store_true',
                        help='enable domain randomization')
    parser.add_argument('--frame-skip', default=1, type=int,
                        help='number of frames to skip')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    args = parser.parse_args()
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
    env.reset()
    env.render()

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print('RESET')
            env.reset()
            env.render()
        elif symbol == (key.PAGEUP or key.SEMICOLON):
            env.unwrapped.cam_angle[0] = 0
        elif symbol == key.ESCAPE:
            env.close()
            sys.exit(0)
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)

    def update(dt):
        action = np.array([0.0, 0.0])
        if key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0.35, -1])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])
        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5
        obs, reward, done, info = env.step(action)
        #  print('obs', obs.shape)
        print('step_count = %s, reward=%.3f' %
              (env.unwrapped.step_count, reward))

        if key_handler[key.RETURN]:
            from PIL import Image
            im = Image.fromarray(obs)
            im.save('screen.png')

        if done:
            print('done!')
            env.reset()
            env.render()

        env.render()

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

    # Enter main event loop
    pyglet.app.run()
    env.close()
