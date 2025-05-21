import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json

def plot_training_results():
    # Load tensorboard data
    event_acc = EventAccumulator('./tensorboard/train')
    event_acc.Reload()

    # Extract metrics
    train_rewards = [(s.step, s.value) for s in event_acc.Scalars('episode_reward')]
    eval_rewards = [(s.step, s.value) for s in event_acc.Scalars('eval_reward')]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot data
    train_data = np.array(train_rewards)
    eval_data = np.array(eval_rewards)
    
    plt.plot(train_data[:,0], train_data[:,1], 
             label='Training Reward', alpha=0.6)
    plt.plot(eval_data[:,0], eval_data[:,1], 
             label='Evaluation Reward', alpha=0.8)
    
    plt.title('CarRacing-v0 Learning Curves')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Save plot
    plt.savefig('./results/learning_curves.png')
    print("Learning curves saved to ./results/learning_curves.png")

    # Print statistics
    print("\nTraining Statistics:")
    print(f"Final training reward: {train_data[-1,1]:.2f}")
    print(f"Best evaluation reward: {max(eval_data[:,1]):.2f}")

if __name__ == "__main__":
    plot_training_results()
