import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import gym
from train_carracing import run_episode
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
import json
from datetime import datetime
import imageio

def load_tensorboard_data(tensorboard_path):
    """Load tensorboard data from event files"""
    event_acc = EventAccumulator(tensorboard_path)
    event_acc.Reload()
    
    # Extract training data
    train_rewards = [(s.step, s.value) for s in event_acc.Scalars('episode_reward')]
    eval_rewards = [(s.step, s.value) for s in event_acc.Scalars('eval_reward')]
    
    return np.array(train_rewards), np.array(eval_rewards)

def plot_learning_curves(train_data, eval_data, save_path):
    """Create and save learning curves plot"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot training rewards
    plt.plot(train_data[:,0], train_data[:,1], label='Training', alpha=0.6)
    
    # Plot evaluation rewards
    plt.plot(eval_data[:,0], eval_data[:,1], label='Evaluation', alpha=0.6)
    
    plt.title('CarRacing-v0 Learning Curves')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def evaluate_agent(env, agent, n_episodes=15, save_video=True):
    """Evaluate agent performance and optionally save a video"""
    rewards = []
    
    if save_video:
        video_frames = []
    
    for i in range(n_episodes):
        print(f"Evaluation episode {i+1}/{n_episodes}")
        stats = run_episode(
            env=env,
            agent=agent,
            deterministic=True,
            do_training=False,
            rendering=True
        )
        rewards.append(stats.episode_reward)
        
    return np.mean(rewards), np.std(rewards)

if __name__ == "__main__":
    # Load tensorboard data
    tensorboard_path = "./tensorboard/train"
    train_data, eval_data = load_tensorboard_data(tensorboard_path)
    
    # Create plots
    plot_learning_curves(train_data, eval_data, "./results/learning_curves.png")
    
    # Load and evaluate best model
    env = gym.make("CarRacing-v0").unwrapped
    
    state_dim = (96, 96, 1)
    num_actions = 5
    
    Q_net = CNN(input_shape=state_dim, num_actions=num_actions)
    Q_target_net = CNN(input_shape=state_dim, num_actions=num_actions)
    
    agent = DQNAgent(
        Q=Q_net,
        Q_target=Q_target_net,
        num_actions=num_actions
    )
    
    # Load best model
    agent.load("./models_carracing/dqn_agent.pt")
    
    # Evaluate agent
    mean_reward, std_reward = evaluate_agent(env, agent, n_episodes=15)
    
    # Save evaluation results
    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "hyperparameters": {
            "gamma": 0.95,
            "batch_size": 32,
            "epsilon": 0.3,
            "tau": 0.01,
            "learning_rate": 1e-4,
            "skip_frames": 2
        }
    }
    
    with open("./results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("Results saved to ./results/")
