import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import Evaluation
from agent.networks import MLP
from utils import EpisodeStats
from torch.utils.tensorboard import SummaryWriter


def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    model_dir="./models_cartpole",
    tensorboard_dir="./tensorboard",
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    print("... train agent")

    # Add eval_reward to the stats list
    tensorboard = Evaluation(
        tensorboard_dir, "train", ["episode_reward", "a_0", "a_1", "eval_reward"]
    )

    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "a_0": stats.get_action_usage(0),
                "a_1": stats.get_action_usage(1),
            },
        )

        # Evaluate agent
        if i % eval_cycle == 0:
            eval_rewards = []
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False)
                eval_rewards.append(eval_stats.episode_reward)
            tensorboard.write_episode_data(i, eval_dict={"eval_reward": np.mean(eval_rewards)})

        # store model
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


def evaluate_agent(env, agent, n_episodes=5):
    eval_rewards = []
    for _ in range(n_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False)
        eval_rewards.append(stats.episode_reward)
    return np.mean(eval_rewards)


if __name__ == "__main__":

    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 20  # evaluate every 10 episodes

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)
    hidden_dim = 400
    Q_net = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dim=hidden_dim)
    Q_target_net = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dim=hidden_dim)
    
    agent = DQNAgent(
        Q=Q_net, 
        Q_target=Q_target_net, 
        num_actions=num_actions,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-4
    )

    # Create tensorboard writer
    writer = SummaryWriter('./tensorboard_logs/cartpole_training')

    # Start training
    n_episodes = 200
    eval_frequency = 20

    for episode in range(n_episodes):
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        
        # Log training reward (with exploration)
        writer.add_scalar('Train/Episode_Reward', stats.episode_reward, episode)
        
        # Evaluate agent every eval_frequency episodes
        if episode % eval_frequency == 0:
            eval_reward = evaluate_agent(env, agent)
            writer.add_scalar('Eval/Mean_Episode_Reward', eval_reward, episode)
            print(f"Episode {episode}, Eval reward: {eval_reward:.2f}")
    
    # Save the trained agent
    if not os.path.exists("./models_cartpole"):
        os.mkdir("./models_cartpole")
    agent.save("./models_cartpole/dqn_agent.pt")
    
    env.close()
    writer.close()
