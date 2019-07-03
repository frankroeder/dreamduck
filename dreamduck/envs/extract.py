import numpy as np
import random
import os
import config
import gym
from model import make_model

MAX_FRAMES = 150
MAX_TRIALS = 4000
MIN_LENGTH = 50

render_mode = False  # for debugging.

full_episode = True

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

model = make_model(config.games['duckietown'])

total_frames = 0
model.make_env(render_mode=render_mode, full_episode=full_episode)
for trial in range(MAX_TRIALS):
    try:
        random_generated_int = random.randint(0, 2**31-1)
        filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
        recording_obs = []
        recording_action = []

        recording_restart = []
        recording_reward = []

        np.random.seed(random_generated_int)
        model.env.seed(random_generated_int)

        obs = model.env.reset()

        if obs is None:
            obs = np.zeros(model.input_size)

        for frame in range(MAX_FRAMES):
            if render_mode:
                model.env.render("human")
            action = model.get_action(obs)  # use more diverse random policy:
            recording_obs.append(obs)
            recording_action.append(action)
            obs, reward, done, info = model.env.step(action)

            recording_restart.append(done)
            recording_reward.append(reward)

            if (not full_episode) and done:
                break

        total_frames += frame
        print("dead at", frame, "total recorded frames for this worker",
              total_frames)
        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.float16)
        recording_restart = np.array(recording_restart, dtype=np.uint8)
        recording_reward = np.array(recording_reward, dtype=np.float16)

        if (len(recording_obs) > MIN_LENGTH):
            np.savez_compressed(filename, obs=recording_obs,
                                action=recording_action,
                                restart=recording_restart,
                                reward=recording_reward)
    except gym.error.Error:
        print("stupid doom error, life goes on")
        model.env.close()
        model.make_env(render_mode=render_mode)
        continue
model.env.close()
