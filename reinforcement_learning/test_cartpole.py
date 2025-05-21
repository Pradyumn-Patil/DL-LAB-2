import os
from datetime import datetime
import gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter

np.random.seed(0)

if __name__ == "__main__":
    # Create tensorboard writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'./tensorboard_logs/test_cartpole/{current_time}'
    writer = SummaryWriter(log_dir)

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2
    hidden_dim = 400

    # Load networks and create agent
    Q_net = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dim=hidden_dim)
    Q_target_net = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dim=hidden_dim)
    agent = DQNAgent(Q=Q_net, Q_target=Q_target_net, num_actions=num_actions)
    
    # Load saved model
    agent.load("./models_cartpole/dqn_agent.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent, deterministic=True, do_training=False, rendering=True
        )
        episode_rewards.append(stats.episode_reward)
        # Log episode reward to tensorboard
        writer.add_scalar('Test/Episode_Reward', stats.episode_reward, i)

    # Calculate and log final statistics
    mean_reward = np.array(episode_rewards).mean()
    std_reward = np.array(episode_rewards).std()
    writer.add_scalar('Test/Mean_Reward', mean_reward, 0)
    writer.add_scalar('Test/Std_Reward', std_reward, 0)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    writer.close()
    print("... finished")
    print(f"TensorBoard logs saved to {log_dir}")
    print("To view results, run: tensorboard --logdir=./tensorboard_logs")
