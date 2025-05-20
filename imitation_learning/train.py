import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import sys

# Add parent directory to sys.path so utils.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):
    # 1. Convert images to grayscale
    X_train = np.array([utils.rgb2gray(img) for img in X_train])
    X_valid = np.array([utils.rgb2gray(img) for img in X_valid])
    # 2. Discretize actions
    y_train = np.array([utils.action_to_id(a) for a in y_train])
    y_valid = np.array([utils.action_to_id(a) for a in y_valid])
    # 3. Add channel dimension
    X_train = X_train[..., np.newaxis]  # (N, 96, 96, 1)
    X_valid = X_valid[..., np.newaxis]
    # 4. Transpose to (N, 1, 96, 96) for PyTorch
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_valid = np.transpose(X_valid, (0, 3, 1, 2))
    return X_train, y_train, X_valid, y_valid


def sample_minibatch(X, y, batch_size):
    idx = np.random.choice(len(X), batch_size, replace=False)
    return X[idx], y[idx]


def compute_accuracy(agent, X, y, batch_size=128):
    n = len(X)
    correct = 0
    total = 0
    print(f"Computing accuracy on {n} samples")
    
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        X_batch = X[i:batch_end]
        y_batch = y[i:batch_end]
        preds = agent.predict(X_batch)
        # Fix: Add argmax to get class predictions from logits
        pred_classes = np.argmax(preds, axis=1)
        correct += np.sum(pred_classes == y_batch)
        total += len(y_batch)
        if i % 1000 == 0:
            print(f"Processed {i}/{n} samples...")
            
    acc = correct / total
    print(f"Final accuracy: {acc:.3f}")
    return acc

def compute_loss_and_accuracy(agent, X, y, batch_size=128):
    n = len(X)
    correct = 0
    total = 0
    total_loss = 0
    
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        X_batch = X[i:batch_end]
        y_batch = y[i:batch_end]
        
        # Compute loss
        X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=agent.device)
        y_tensor = torch.tensor(y_batch, dtype=torch.long, device=agent.device)
        logits = agent.net(X_tensor)
        loss = agent.loss_fn(logits, y_tensor)
        total_loss += loss.item() * len(y_batch)
        
        # Compute accuracy
        preds = agent.predict(X_batch)
        correct += np.sum(np.argmax(preds, axis=1) == y_batch)
        total += len(y_batch)
    
    return total_loss / total, correct / total


def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_minibatches,
    batch_size,
    lr,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):
    # Clear old tensorboard files
    if os.path.exists(tensorboard_dir):
        for file in os.listdir(tensorboard_dir):
            file_path = os.path.join(tensorboard_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # Create directories if they don't exist
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    
    print("... train model")
    agent = BCAgent(input_channels=1, num_actions=5, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(agent.optimizer, 'min', patience=5, factor=0.5)
    best_valid_loss = float('inf')
    
    # Initialize tensorboard
    stats = ["train_loss", "train_acc", "valid_loss", "valid_acc", "learning_rate"]
    tensorboard_eval = Evaluation(tensorboard_dir, "imitation_learning", stats)
    
    for i in range(n_minibatches):
        X_batch, y_batch = sample_minibatch(X_train, y_train, batch_size)
        train_loss = agent.update(X_batch, y_batch)
        
        if i % 10 == 0:
            # Compute metrics
            train_acc = compute_accuracy(agent, X_batch, y_batch)
            valid_loss, valid_acc = compute_loss_and_accuracy(agent, X_valid, y_valid)
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                agent.save(os.path.join(model_dir, "best_agent.pt"))
            
            tensorboard_eval.write_episode_data(
                i,
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                    "learning_rate": current_lr
                },
            )
            
            # Update learning rate scheduler
            scheduler.step(valid_loss)
            
            print(
                f"Step {i}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.3f}, lr={current_lr:.6f}"
            )
    
    # Load best model before returning
    agent.load(os.path.join(model_dir, "best_agent.pt"))
    print("Training completed. Best model loaded.")
    
    # Save final model
    agent.save(os.path.join(model_dir, "final_agent.pt"))


if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=1
    )

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=1e-4)
