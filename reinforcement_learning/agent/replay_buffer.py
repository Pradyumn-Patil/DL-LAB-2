import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.terminals = []
        self.idx = 0

    def add_transition(self, state, action, next_state, reward, terminal):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)
            self.rewards.append(reward)
            self.terminals.append(terminal)
        else:
            idx = self.idx % self.capacity
            self.states[idx] = state
            self.actions[idx] = action
            self.next_states[idx] = next_state
            self.rewards[idx] = reward
            self.terminals[idx] = terminal
            self.idx += 1

    def next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self.states), batch_size)
        batch_states = np.array([self.states[i] for i in batch_indices])
        batch_actions = np.array([self.actions[i] for i in batch_indices])
        batch_next_states = np.array([self.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self.rewards[i] for i in batch_indices])
        batch_terminals = np.array([self.terminals[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminals

    def size(self):
        return len(self.states)
