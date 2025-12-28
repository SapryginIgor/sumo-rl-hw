"""Q-learning Agent class."""


import numpy as np

class MyQLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, eps=0.0, eps_decay=1.0, seed=42):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: np.zeros(action_space.n)}
        self.acc_reward = 0
        self.T_decay=0
        self.eps = eps
        self.eps_decay=eps_decay
        self.rng = np.random.default_rng(seed)


    def act(self):
        """Choose action based on Q-table."""
        eps = self.eps * self.eps_decay**self.T_decay
        if eps > 0:
          if self.rng.random() < eps:
            self.action = self.rng.integers(0, self.action_space.n)
          else:
            self.action = np.argmax(self.q_table[self.state])
        else:
            self.action = np.argmax(self.q_table[self.state])
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space.n)

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward