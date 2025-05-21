# export DISPLAY=:0

import sys
import os
import time  # Add this import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray, LEFT, RIGHT, STRAIGHT, ACCELERATE, BRAKE
from agent.dqn_agent import DQNAgent
from agent.networks import CNN  # Add this import


def run_episode(
    env,
    agent,
    deterministic,
    skip_frames=0,
    do_training=True,
    rendering=True,
    max_timesteps=1000,
    history_length=0,
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0

    # Simplified render setup
    state = env.reset()
    env.render()
    time.sleep(0.5)

    # fix bug of corrupted states without rendering in gym environment
    if hasattr(env, 'viewer') and env.viewer is not None:
        env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)

    while True:
        # Modified action sampling for better exploration
        if not deterministic and np.random.random() < agent.epsilon:
            # Biased random action selection:
            # Higher probability for acceleration and going straight
            action_probs = [0.3,    # STRAIGHT - 30%
                            0.15,    # LEFT - 15%
                            0.15,    # RIGHT - 15%
                            0.3,     # ACCELERATE - 30%
                            0.1]     # BRAKE - 10%
            action_id = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_id = agent.act(state=state, deterministic=True)
        # Map action_id to actual action
        actions = [
            [0.0, 0.0, 0.0],    # STRAIGHT
            [-1.0, 0.0, 0.0],   # LEFT
            [1.0, 0.0, 0.0],    # RIGHT
            [0.0, 1.0, 0.0],    # ACCELERATE
            [0.0, 0.0, 0.8],    # BRAKE
        ]
        action = actions[action_id]

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()
                time.sleep(0.05)
            
            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    history_length=0,
    model_dir="./models_carracing",
    tensorboard_dir="./tensorboard",
):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    print("... train agent")
    
    # Fix: Create proper tensorboard directory path
    tensorboard = Evaluation(
        tensorboard_dir,
        "train",
        ["episode_reward", "straight", "left", "right", "accel", "brake", 
         "eval_reward", "epsilon", "avg_q_value", "loss"]  # Added more metrics
    )

    # Define max_timesteps (was undefined)
    max_timesteps = 1000  # You can adjust this value

    for i in range(num_episodes):
        print("episode %d" % i)

        stats = run_episode(
            env,
            agent,
            deterministic=False,
            skip_frames=2,  # Added frame skipping
            do_training=True,
            max_timesteps=500 if i < 100 else 1000,  # Shorter episodes initially
            history_length=history_length
        )

        # Log additional training metrics
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "straight": stats.get_action_usage(STRAIGHT),
                "left": stats.get_action_usage(LEFT),
                "right": stats.get_action_usage(RIGHT),
                "accel": stats.get_action_usage(ACCELERATE),
                "brake": stats.get_action_usage(BRAKE),
                "epsilon": agent.epsilon,
                "avg_q_value": agent.get_q_values(stats.last_state).mean().item() if hasattr(stats, 'last_state') else 0,
                "loss": agent.loss if hasattr(agent, 'loss') else 0
            },
        )

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...
        # Evaluate agent
        if i % eval_cycle == 0:
            eval_rewards = []
            for j in range(num_eval_episodes):
                eval_stats = run_episode(
                    env,
                    agent,
                    deterministic=True,
                    do_training=False,
                    history_length=history_length
                )
                eval_rewards.append(eval_stats.episode_reward)
            # Log evaluation metrics
            tensorboard.write_episode_data(
                i,
                eval_dict={"eval_reward": np.mean(eval_rewards)}
            )

        # store model
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":

    # Set up display for Linux
    if sys.platform == "linux":
        os.environ['SDL_VIDEODRIVER'] = 'x11'

    num_eval_episodes = 5
    eval_cycle = 20
    history_length = 0

    # Create environment without render_mode
    env = gym.make("CarRacing-v0").unwrapped
    
    # Initial render setup
    env.reset()
    env.render()
    time.sleep(1.0)

    # Define network parameters
    state_dim = (96, 96, history_length + 1)  # Input image dimensions
    num_actions = 5  # [STRAIGHT, LEFT, RIGHT, ACCELERATE, BRAKE]
    
    # Initialize CNN networks
    Q_net = CNN(input_shape=state_dim, num_actions=num_actions)
    Q_target_net = CNN(input_shape=state_dim, num_actions=num_actions)
    
    # Initialize agent with higher epsilon for better exploration
    agent = DQNAgent(
        Q=Q_net,
        Q_target=Q_target_net,
        num_actions=num_actions,
        gamma=0.95,
        batch_size=32,
        epsilon=0.3,  # Increased epsilon for better exploration
        tau=0.01,
        lr=1e-4,
        history_length=history_length
    )

    # Start training
    train_online(
        env=env,
        agent=agent,
        num_episodes=1000,
        history_length=history_length,
        model_dir="./models_carracing"
    )
