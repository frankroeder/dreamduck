import numpy as np
from cv2 import resize

SCREEN_X = 64
SCREEN_Y = 64

def _process_frame(frame):
    obs = frame[:, :, :].astype(np.float)/255.0
    obs = np.array(resize(obs, (SCREEN_X, SCREEN_Y)))
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    return obs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


def passthru(x):
    return x


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample(p):
    return np.argmax(np.random.multinomial(1, p))
