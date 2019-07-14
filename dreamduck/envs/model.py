import numpy as np
import json
import os
from env import make_env
from dreamduck.envs.rnn.rnn import hps_sample, MDNRNN, rnn_init_state, \
    rnn_next_state, rnn_output, rnn_output_size, rnn_model_path_name
from dreamduck.envs.vae.vae import ConvVAE, vae_model_path_name

render_mode = True
RENDER_DELAY = False

# controls whether we concatenate (z, c, h)
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3  # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH


def make_model(load_model=True):
    model = Model(load_model=load_model)
    return model


class Model:
    def __init__(self, load_model=True):
        self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False,
                           reuse=True)
        self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

        if load_model:
            self.vae.load_json(os.path.join(vae_model_path_name, 'vae.json'))
            self.rnn.load_json(os.path.join(rnn_model_path_name, 'rnn.json'))

        self.state = rnn_init_state(self.rnn)
        self.rnn_mode = True
        self.input_size = rnn_output_size(EXP_MODE)
        self.z_size = 64

        if EXP_MODE == MODE_Z_HIDDEN:  # one hidden layer
            self.hidden_size = 512
            self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
            self.bias_hidden = np.random.randn(self.hidden_size)
            self.weight_output = np.random.randn(self.hidden_size, 2)
            self.bias_output = np.random.randn(2)
            self.param_count = ((self.input_size+1)*self.hidden_size) + \
                                (self.hidden_size*2+2)
        else:
            self.weight = np.random.randn(self.input_size, 2)
            self.bias = np.random.randn(2)
            self.param_count = (self.input_size)*2+2

        self.render_mode = False

    def make_env(self, seed=-1, render_mode=False, load_model=True,
                 full_episode=True):
        self.render_mode = render_mode
        self.env = make_env(seed=seed, render_mode=render_mode,
                            load_model=load_model, full_episode=full_episode)

    def reset(self):
        self.state = rnn_init_state(self.rnn)

    def encode_obs(self, obs):
        # convert raw obs to z, mu, logvar
        result = np.copy(obs).astype(np.float)/255.0
        result = result.reshape(1, 64, 64, 3)
        mu, logvar = self.vae.encode_mu_logvar(result)
        mu = mu[0]
        logvar = logvar[0]
        s = logvar.shape
        z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
        return z, mu, logvar

    def get_action(self, z):
        print(self.state.h[0].shape, z.shape)
        h = rnn_output(self.state, z, EXP_MODE)
        if EXP_MODE == MODE_Z_HIDDEN:
            h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
            action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
        else:
            action = np.tanh(np.dot(h, self.weight) + self.bias)

        self.state = rnn_next_state(self.rnn, z, action, self.state)
        return action

    def set_model_params(self, model_params):
        if EXP_MODE == MODE_Z_HIDDEN:  # one hidden layer
            params = np.array(model_params)
            cut_off = (self.input_size+1)*self.hidden_size
            params_1 = params[:cut_off]
            params_2 = params[cut_off:]
            self.bias_hidden = params_1[:self.hidden_size]
            self.weight_hidden = params_1[self.hidden_size:].reshape(
                self.input_size, self.hidden_size)
            self.bias_output = params_2[:2]
            self.weight_output = params_2[2:].reshape(self.hidden_size, 2)
        else:
            self.bias = np.array(model_params[:2])
            self.weight = np.array(
                model_params[2:]).reshape(self.input_size, 2)

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return np.random.standard_cauchy(self.param_count)*stdev

    def init_random_model_params(self, stdev=0.1):
        params = self.get_random_model_params(stdev=stdev)
        self.set_model_params(params)
        vae_params = self.vae.get_random_model_params(stdev=stdev)
        self.vae.set_model_params(vae_params)
        rnn_params = self.rnn.get_random_model_params(stdev=stdev)
        self.rnn.set_model_params(rnn_params)
