import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer

# Add device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-4,
        history_length=0,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.to(device)
        self.Q_target = Q_target.to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer()  # Remove history_length parameter since it's not used

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        # Store transition in replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        # Only train if enough samples in replay buffer
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample batch from replay buffer
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminals = self.replay_buffer.next_batch(
            self.batch_size
        )

        # Convert numpy arrays to torch tensors
        states = torch.FloatTensor(batch_states).to(device)
        actions = torch.LongTensor(batch_actions).to(device)
        next_states = torch.FloatTensor(batch_next_states).to(device)
        rewards = torch.FloatTensor(batch_rewards).to(device)
        terminals = torch.FloatTensor(batch_terminals).to(device)

        # Compute TD targets
        next_q_values = self.Q_target(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        td_targets = rewards + (1 - terminals) * self.gamma * max_next_q_values

        # Get Q values for the actions taken
        q_values = self.Q(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss and update Q network
        loss = self.loss_function(q_values, td_targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if deterministic or np.random.rand() > self.epsilon:
            with torch.no_grad():
                q_values = self.Q(state)
                action_id = q_values.max(1)[1].item()
        else:
            action_id = np.random.randint(self.num_actions)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))

    def get_q_values(self, state):
        """Returns Q-values for a given state"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.Q(state)
