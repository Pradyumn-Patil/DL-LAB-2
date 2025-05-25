from datetime import datetime
import os
import json
import gym
import numpy as np
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import CNN

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length = 0  # Set this to match your training

    # Define networks and load agent
    state_dim = (96, 96, history_length + 1)
    num_actions = 5  # [STRAIGHT, LEFT, RIGHT, ACCELERATE, BRAKE]
    Q_net = CNN(input_shape=state_dim, num_actions=num_actions)
    Q_target_net = CNN(input_shape=state_dim, num_actions=num_actions)
    agent = DQNAgent(
        Q=Q_net,
        Q_target=Q_target_net,
        num_actions=num_actions,
        gamma=0.95,
        batch_size=32,
        epsilon=0.0,  # Greedy for testing
        tau=0.01,
        lr=1e-4,
        history_length=history_length
    )
    agent.load("./models_carracing/dqn_agent.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent, deterministic=True, do_training=False, rendering=True, history_length=history_length
        )
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print("... finished")