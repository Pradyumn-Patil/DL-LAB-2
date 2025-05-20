import torch
import torch.nn as nn
import torch.optim as optim
from agent.networks import CNN
import torch.nn.functional as F


class BCAgent:

    def __init__(self, input_channels=1, num_actions=5, lr=1e-4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CNN(input_channels=input_channels, num_actions=num_actions).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch):
        # X_batch: [batch, 1, 96, 96], y_batch: [batch] (action ids)
        X_batch = torch.tensor(X_batch, dtype=torch.float32, device=self.device)
        y_batch = torch.tensor(y_batch, dtype=torch.long, device=self.device)
        self.optimizer.zero_grad()
        logits = self.net(X_batch)
        loss = self.loss_fn(logits, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, X):
        # X: [batch, 1, 96, 96]
        self.net.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.net(X)
            # Return softmax probabilities instead of just argmax
            probs = F.softmax(logits, dim=1)
        self.net.train()
        return probs.cpu().numpy()

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name, map_location=self.device))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
