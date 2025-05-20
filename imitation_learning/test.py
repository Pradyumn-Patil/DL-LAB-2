import sys

sys.path.append(".")
from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0

    state = env.reset()

    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()

    while True:
        # Preprocess state - same as in training
        state_gray = rgb2gray(state)
        state_proc = np.expand_dims(state_gray, axis=(0, 1))  # Add batch and channel dims

        # Get action probabilities and convert to continuous action
        action_probs = agent.predict(state_proc)[0]
        # Get the most confident action
        action_id = np.argmax(action_probs)
        # Convert to continuous action but scale by confidence
        base_action = id_to_action(action_id)
        confidence = action_probs[action_id]
        a = base_action * confidence  # Scale action by confidence

        # Clip acceleration to prevent spinning out
        a[1] = np.clip(a[1], 0, 0.8)  # Clip acceleration between 0 and 0.8

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test

    # Load agent
    agent = BCAgent(input_channels=1, num_actions=5)
    agent.load("./models/agent.pt")

    env = gym.make("CarRacing-v0").unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print("... finished")
