import numpy as np

class GradientBandit:
    def __init__(self, num_actions, alpha=0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = np.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0

    def softmax(self):
        exp_pref = np.exp(self.preferences - np.max(self.preferences))
        return exp_pref / np.sum(exp_pref)
    
    def select_action(self):
        probs = self.softmax()
        return np.random.choice(self.num_actions, p=probs)

    def update(self, action, reward):
        self.time += 1
        self.avg_reward += (reward - self.avg_reward) / self.time

        probs = self.softmax()

        for a in range(self.num_actions):
            if a == action:
                self.preferences[a] += self.alpha * (reward - self.avg_reward) * (1 - probs[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.avg_reward) * probs[a]