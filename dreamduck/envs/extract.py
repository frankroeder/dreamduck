import numpy as np
import random
import os
import gym
from model import make_model
import argparse

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)


def generate_records(max_frames, min_frames, max_trials, render_mode=False,
                     full_episode=False):

    model = make_model()
    total_frames = 0
    model.make_env(render_mode=render_mode, full_episode=full_episode,
                   load_model=False)

    for trial in range(max_trials):
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

            for frame in range(max_frames):
                if render_mode:
                    model.env.render("human")

                z, mu, logvar = model.encode_obs(obs)
                action = model.get_action(z)
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

            if (len(recording_obs) > min_frames):
                np.savez_compressed(filename, obs=recording_obs,
                                    action=recording_action,
                                    restart=recording_restart,
                                    reward=recording_reward)
        except gym.error.Error:
            print("duckietown error")
            model.env.close()
            model.make_env(render_mode=render_mode)
            continue
    model.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true',
                        help='Render rollouts')
    parser.add_argument('--full-episode', action='store_true',
                        help='')
    parser.add_argument('--max-frames', default=150, type=int,
                        help='Control uncertainty')
    parser.add_argument('--max-trials', default=4000, type=int,
                        help='Control uncertainty')
    parser.add_argument('--min-frames', default=50, type=int,
                        help='Control uncertainty')

    args = parser.parse_args()
    render_mode = args.render
    full_episode = args.full_episode
    max_frames = args.max_frames
    min_frames = args.min_frames
    max_trials = args.max_trials
    generate_records(max_frames, min_frames, max_trials,
                     render_mode=render_mode, full_episode=full_episode)


if __name__ == "__main__":
    main()
