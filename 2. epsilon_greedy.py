import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm

BANDITS_PROBOBALITIES = [0.25, 0.5, 0.75]
epsilon = 0.5
NB_RUNS = 1000

class Bandit():
    def __init__(self, p):
        self.p = p
        self.N = 0
        self.estimate_p = 0

    def pull(self):
        if np.random.rand() < self.p:
            reward = 1
        else:
            reward = 0
        self.update(reward)
        return reward

    def update(self, reward):
        self.N += 1
        self.estimate_p = self.estimate_p + 1/self.N * (reward - self.estimate_p)

if __name__ == '__main__':
    bandits = [Bandit(p) for p in BANDITS_PROBOBALITIES]
    nb_optimal_choice = 0
    nb_non_optimal_choice = 0
    rewards = []
    for i in range(NB_RUNS):
        decaying_eps = epsilon / 1.1**(i)
        if np.random.rand() > decaying_eps:
            nb_optimal_choice += 1
            j = np.argmax([bandit.estimate_p for bandit in bandits])
        else:
            j = np.random.randint(0, len(bandits))
        rewards.append(bandits[j].pull())

    #print('estimates:', [b.estimate_p for b in bandits])
    win_rates = np.cumsum(rewards)/(np.arange(NB_RUNS)+1)
    plt.plot(win_rates)
    plt.plot(NB_RUNS * [np.max(BANDITS_PROBOBALITIES)])
    plt.show()
