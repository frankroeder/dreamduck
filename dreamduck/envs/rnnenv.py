import numpy as np
import json
import sys
import argparse
from gym import spaces
from gym.spaces.box import Box
from dreamduck.envs.rnn.rnn import get_pi_idx, hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size
from dreamduck.envs.vae.vae import ConvVAE
import os
from gym.utils import seeding
import tensorflow as tf
import gym

SCREEN_X = 64
SCREEN_Y = 64
TEMPERATURE = 1.25
# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3
MODE_ZH = 4

EXP_MODE = MODE_ZH

with open(os.path.join('initial_z', 'initial_z.json'), 'r') as f:
    [initial_mu, initial_logvar] = json.load(f)

initial_mu_logvar = [list(elem) for elem in zip(initial_mu, initial_logvar)]

model_path_name = 'dreamduck/envs/tf_initial_z'

# Dreaming


class DuckieTownRNN(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, agent, load_model=True):
        self.env_name = "carracing"
        self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

        self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

        if load_model:
            self.vae.load_json('vae/vae.json')
            self.rnn.load_json('rnn/rnn.json')

        self.state = rnn_init_state(self.rnn)
        self.rnn_mode = True

        self.input_size = rnn_output_size(EXP_MODE)
        self.z_size = 32

        if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
            self.hidden_size = 40
            self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
            self.bias_hidden = np.random.randn(self.hidden_size)
            self.weight_output = np.random.randn(self.hidden_size, 3)
            self.bias_output = np.random.randn(3)
            self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*3+3)
        else:
            self.weight = np.random.randn(self.input_size, 3)
            self.bias = np.random.randn(3)
            self.param_count = (self.input_size)*3+3

        self.render_mode = False
        self.observation_space = Box(low=-50., high=50., shape=(64))
        self._seed()
        self.agent = agent
        self.vae = agent.vae
        self.rnn = agent.rnn
        self.z_size = self.rnn.hps.output_seq_width
        self.viewer = None
        self.frame_count = None
        self.z = None
        self.temperature = 0.7
        self.vae_frame = None
        self._reset()

    def _sample_z(self, mu, logvar):
        z = mu + np.exp(logvar/2.0) * self.np_random.randn(*logvar.shape)
        return z

    def _reset(self):
        idx = self.np_random.randint(0, len(initial_mu_logvar))
        init_mu, init_logvar = initial_mu_logvar[idx]
        init_mu = np.array(init_mu)/10000.
        init_logvar = np.array(init_logvar)/10000.
        self.z = self._sample_z(init_mu, init_logvar)
        self.frame_count = 0
        return self.z

    def _seed(self, seed=None):
        if seed:
            tf.set_random_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sample_next_z(self, action):
        s_model = self.rnn
        temperature = self.temperature

        sess = s_model.sess
        hps = s_model.hps

        OUTWIDTH = hps.output_seq_width

        prev_x = np.zeros((1, 1, OUTWIDTH))
        prev_x[0][0] = self.z

        strokes = np.zeros((1, OUTWIDTH), dtype=np.float32)

        input_x = np.concatenate((prev_x, action.reshape(1, 1, 3)), axis=2)
        feed = {s_model.input_x: input_x,
                s_model.initial_state: self.agent.state}
        [logmix, mean, logstd, self.agent.state] = sess.run(
            [s_model.out_logmix, s_model.out_mean, s_model.out_logstd, s_model.final_state], feed)

        # adjust temperatures
        logmix2 = np.copy(logmix)/temperature
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

        mixture_idx = np.zeros(OUTWIDTH)
        chosen_mean = np.zeros(OUTWIDTH)
        chosen_logstd = np.zeros(OUTWIDTH)
        for j in range(OUTWIDTH):
            idx = get_pi_idx(self.np_random.rand(), logmix2[j])
            mixture_idx[j] = idx
            chosen_mean[j] = mean[j][idx]
            chosen_logstd[j] = logstd[j][idx]

        rand_gaussian = self.np_random.randn(OUTWIDTH)*np.sqrt(temperature)
        next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

        next_z = next_x.reshape(OUTWIDTH)

        return next_z

    def _step(self, action):
        self.frame_count += 1
        next_z = self._sample_next_z(action)
        reward = 0
        done = False
        if self.frame_count > 1200:
            done = True
        self.z = next_z
        return next_z, reward, done, {}

    def decode_obs(self, z):
        # decode the latent vector
        img = self.vae.decode(z.reshape(1, self.z_size)) * 255.
        img = np.round(img).astype(np.uint8)
        img = img.reshape(64, 64, 3)
        return img

    def _render(self, mode='human', close=False):
        if not self.render_mode:
            return

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            img = self._get_image(upsize=True)
            return img
        elif mode == 'human':
            img = self._get_image(upsize=True)
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)


if __name__ == "__main__":

    env = DuckieTownRNN(render_mode=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', default=.01, type=float,
                        help='Control uncertainty')
    args = parser.parse_args()
    TEMPERATURE = args.temp
    if env.render_mode:
        from pyglet.window import key
    action = np.array([0.0, 0.0])
    overwrite = False

    def key_press(k, mod):
        global action
        if k == key.UP:
            action = np.array([0.44, 0.0])
        if k == key.DOWN:
            action = np.array([-0.44, 0])
        if k == key.LEFT:
            action = np.array([0.35, +1])
        if k == key.RIGHT:
            action = np.array([0.35, -1])
        if k == key.SPACE:
            action = np.array([0, 0])
        if k == key.ESCAPE:
            env.close()
            sys.exit(0)

    def key_release(k, mod):
        action[0] = 0.
        action[1] = 0.

    env._render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    reward_list = []

    for i in range(400):
        env._reset()
        total_reward = 0.0
        repeat = np.random.randint(1, 11)
        obs = env._reset()

        while True:
            obs, reward, done, info = env._step(action)
            total_reward += reward

            if env.render_mode:
                env._render()
            if done:
                break
        reward_list.append(total_reward)
        print('cumulative reward', total_reward)
    env.close()
    print('average reward', np.mean(reward_list))
