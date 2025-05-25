import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalars(event_acc, tag):
    try:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return np.array(steps), np.array(values)
    except KeyError:
        return np.array([]), np.array([])

def plot_tensorboard_logs(tensorboard_dir="./tensorboard/imitation_learning", save_path="./results/imitation_learning_curves.png"):
    if not os.path.exists(tensorboard_dir):
        print(f"Tensorboard directory {tensorboard_dir} does not exist.")
        return

    # Only use event files inside the specified subfolder (not parent tensorboard dir)
    event_files = [os.path.join(tensorboard_dir, f) for f in os.listdir(tensorboard_dir) if "events.out.tfevents" in f]
    if not event_files:
        print(f"No tensorboard event files found in {tensorboard_dir}")
        return

    # Use the first event file found in the subfolder
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()

    tags = ["train_loss", "valid_loss", "train_acc", "valid_acc"]
    curves = {}
    for tag in tags:
        steps, values = load_scalars(event_acc, tag)
        curves[tag] = (steps, values)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if len(curves["train_loss"][0]) > 0:
        plt.plot(curves["train_loss"][0], curves["train_loss"][1], label="Train Loss")
    if len(curves["valid_loss"][0]) > 0:
        plt.plot(curves["valid_loss"][0], curves["valid_loss"][1], label="Valid Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    if len(curves["train_acc"][0]) > 0:
        plt.plot(curves["train_acc"][0], curves["train_acc"][1], label="Train Acc")
    if len(curves["valid_acc"][0]) > 0:
        plt.plot(curves["valid_acc"][0], curves["valid_acc"][1], label="Valid Acc")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Learning curves saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard_dir", type=str, default="./tensorboard/imitation_learning")
    parser.add_argument("--save_path", type=str, default="./results/imitation_learning_curves.png")
    args = parser.parse_args()
    plot_tensorboard_logs(args.tensorboard_dir, args.save_path)
