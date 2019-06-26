import numpy as np
import cv2
from pyglet.window import key
import sys
import pyglet
from gym import spaces
from gym.spaces.box import Box
from dreamduck.envs.env import DuckieTownWrapper
from dreamduck.envs.rnn.rnn import reset_graph, rnn_model_path_name, \
    model_rnn_size, model_state_space, MDNRNN, hps_sample
from dreamduck.envs.vae.vae import ConvVAE, vae_model_path_name
import os
from scipy.misc import imresize as resize
from gym.utils import seeding
import tensorflow as tf

# actual observation size
SCREEN_X = 64
SCREEN_Y = 64


def _process_frame(frame):
    obs = frame[:, :, :].astype(np.float)/255.0  # REALLY 84?
    obs = np.array(resize(obs, (SCREEN_Y, SCREEN_X)))
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    return obs


# World Model Representation
class DuckieTownReal(DuckieTownWrapper):
    def __init__(self, render_mode=True, load_model=True):
        super(DuckieTownReal, self).__init__()
        self.current_obs = None

        reset_graph()
        self.vae = ConvVAE(batch_size=1, gpu_mode=False,
                           is_training=False, reuse=True)
        self.rnn = MDNRNN(hps_sample, gpu_mode=False)

        if load_model:
            self.vae.load_json(os.path.join(vae_model_path_name, 'vae.json'))
            self.rnn.load_json(os.path.join(rnn_model_path_name, 'rnn.json'))

        # shape right?
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.outwidth = self.rnn.hps.seq_width
        self.obs_size = self.outwidth + model_rnn_size*model_state_space

        self.observation_space = Box(
            low=0, high=255, shape=(SCREEN_Y, SCREEN_X, 3))
        self.actual_observation_space = Box(
            low=-50., high=50., shape=(self.obs_size,))

        self.zero_state = self.rnn.sess.run(self.rnn.zero_state)
        self._seed()

        self.rnn_state = None
        self.z = None
        self.restart = None
        self.frame_count = None
        self.viewer = None
        self._reset()
        self.reward = 0

    def _step(self, action):
        # update states of rnn
        self.frame_count += 1

        prev_z = np.zeros((1, 1, self.outwidth))
        prev_z[0][0] = self.z

        prev_action = np.reshape(action, (1, 1, 2))

        prev_restart = np.ones((1, 1))
        prev_restart[0] = self.restart

        prev_reward = np.ones((1, 1))
        prev_reward[0][0] = self.reward
        # Real stuff from the env to encode it by the world model
        obs, reward, done, _ = super(DuckieTownReal, self).step(action)
        s_model = self.rnn

        feed = {s_model.input_z: prev_z,
                s_model.input_action: prev_action,
                s_model.input_restart: prev_restart,
                s_model.input_reward: prev_reward,
                s_model.initial_state: self.rnn_state
                }

        self.rnn_state = s_model.sess.run(s_model.final_state, feed)

        small_obs = _process_frame(obs)
        self.current_obs = small_obs
        self.z = self._encode(small_obs)
        self.reward = reward

        if done:
            self.restart = 1
        else:
            self.restart = 0

        return self._current_state(), reward, done, {}
        #  return small_obs, reward, done, {}

    def _encode(self, img):
        simple_obs = np.copy(img).astype(np.float)/255.0
        simple_obs = simple_obs.reshape(1, 64, 64, 3)
        mu, logvar = self.vae.encode_mu_logvar(simple_obs)
        return (mu + np.exp(logvar/2.0) * self.np_random.randn(*logvar.shape))[0]

    def _decode(self, z):
        # decode the latent vector
        img = self.vae.decode(z.reshape(1, 64)) * 255.
        img = np.round(img).astype(np.uint8)
        img = img.reshape(64, 64, 3)
        return img

    def _reset(self):
        obs = super(DuckieTownReal, self).reset()
        small_obs = _process_frame(obs)
        self.current_obs = small_obs
        self.rnn_state = self.zero_state
        self.z = self._encode(small_obs)
        self.restart = 1
        self.frame_count = 0
        return self._current_state()

    def _current_state(self):
        if model_state_space == 2:
            return np.concatenate([self.z, self.rnn_state.c.flatten(), self.rnn_state.h.flatten()], axis=0)
        return np.concatenate([self.z, self.rnn_state.h.flatten()], axis=0)

    def _seed(self, seed=None):
        if seed:
            tf.set_random_seed(seed)
            self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                # If we don't None out this reference pyglet becomes unhappy
                self.viewer = None
            return
        try:
            small_img = self.current_obs
            if small_img is None:
                small_img = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
            small_img = cv2.resize(small_img, (64, 64))
            vae_img = self._decode(self.z)
            vae_img = cv2.resize(vae_img, (64, 64))
            all_img = np.concatenate((small_img, vae_img), axis=1)
            img = all_img
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from pyglet import gl, window, image
                if self.window is None:
                    config = gl.Config(double_buffer=False)
                    self.window = window.Window(
                            width=SCREEN_X,
                            height=SCREEN_Y,
                            resizable=False,
                            config=config
                    )

                self.window.clear()
                self.window.switch_to()
                self.window.dispatch_events()

                # Bind the default frame buffer
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

                # Setup orghogonal projection
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glLoadIdentity()
                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glLoadIdentity()
                gl.glOrtho(0, SCREEN_X, 0, SCREEN_Y, 0, 10)

                # Draw the image to the rendering window
                width = img.shape[1]
                height = img.shape[0]
                img = np.ascontiguousarray(np.flip(img, axis=0))
                from ctypes import POINTER
                img_data = image.ImageData(
                        width,
                        height,
                        'RGB',
                        img.ctypes.data_as(POINTER(gl.GLubyte)),
                        pitch=width * 3,
                )
                img_data.blit(
                        0,
                        0,
                        0,
                        width=SCREEN_X,
                        height=SCREEN_Y
                )
        except:
            pass  # Duckietown has been closed


if __name__ == "__main__":
    env = DuckieTownReal()
    env._reset()
    env._render()

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print('RESET')
            env._reset()
            env._render()
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
        obs, reward, done, info = env._step(action)
        print('obs', obs.shape)
        print('step_count = %s, reward=%.3f' %
              (env.unwrapped.step_count, reward))

        if key_handler[key.RETURN]:
            from PIL import Image
            im = Image.fromarray(obs)
            im.save('screen.png')

        if done:
            print('done!')
            env._reset()
            env._render()

        env._render()

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

    # Enter main event loop
    pyglet.app.run()
    env.close()
